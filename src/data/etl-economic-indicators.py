import pandas as pd
import logging
import requests
import time
import os
from retry import retry
from pathlib import Path

# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    from pyspark.sql import SparkSession
    from databricks.feature_store import FeatureStoreClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FREDDataFetcher:
    """Fetches and processes FRED economic data for storage."""

    def __init__(self, api_key, series_file, output_path="data/fred_economic_indicators", batch_size=1):
        """
        Initialize the FRED data fetcher.

        Args:
            api_key (str): FRED API key.
            series_file (str): Path to CSV file containing series IDs.
            output_path (str): Path for storing processed data (CSV locally, Delta Lake in Databricks).
            batch_size (int): Number of series to process per batch.
        """
        self.api_key = api_key
        self.series_file = series_file
        self.output_path = output_path
        self.batch_size = batch_size
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        if IS_DATABRICKS:
            self.spark = SparkSession.builder.appName("FREDDataETL").getOrCreate()
            self.fs = FeatureStoreClient()
            dbutils.fs.mkdirs(self.output_path)
        else:
            self.spark = None
            self.fs = None
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @retry(tries=3, delay=1, backoff=2, exceptions=(requests.exceptions.RequestException,))
    def fetch_series(self, series_id, start_date="2000-01-01", end_date="2025-04-30"):
        """
        Fetch FRED data for a single series.

        Args:
            series_id (str): FRED series ID.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with date and value columns, or None on failure.
        """
        # Determine frequency based on series ID
        quarterly_series = ['GDP', 'GDPC1', 'GFDEBTN']
        daily_series = ['SP500', 'VIXCLS', 'DGS10', 'TB3MS', 'T10Y2Y', 'BAA', 'TEDRATE', 
                        'DCOILWTICO', 'GOLDPMGBD228NLBM']
        long_series = ['DGS10' , 'VIXCLS', 'T10Y2Y']
        frequency = 'q' if series_id in quarterly_series else 'd' if series_id in daily_series else ''
        alt_start_date = "2017-06-01" if series_id in long_series else start_date
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "realtime_start": alt_start_date,
            "realtime_end": end_date,
            "observation_start": alt_start_date,
            "observation_end": end_date,
            # "frequency": frequency,
            "file_type": "json"
        }
        if frequency == 'd':
            params["aggregation_method"] = "avg"  # Aggregate daily to monthly
        
        if series_id == 'SP500':
            params["realtime_start"] = ""
            params["realtime_end"] = ""
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "observations" not in data or not data["observations"]:
                logger.warning(f"No observations for {series_id}")
                return None
                
            df = pd.DataFrame(data["observations"])[["date", "value"]]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"value": series_id}).set_index("date")
            logger.info(f"Fetched {len(df)} records for {series_id}")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return None

    def fetch_batch(self, series_ids, start_date="2000-01-01", end_date="2025-05-01"):
        """
        Fetch data for a batch of series IDs.

        Args:
            series_ids (list): List of FRED series IDs.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Consolidated DataFrame for the batch.
        """
        result_df = None
        for series_id in series_ids:
            df = self.fetch_series(series_id, start_date, end_date)
            if df is not None:
                if result_df is None:
                    result_df = df
                else:
                    result_df = result_df.join(df, how="outer")
            time.sleep(1)  # FRED API: ~60 requests/minute
        if result_df is not None:
            # Resample to monthly, using mean for numerical values
            result_df = result_df.resample("MS").mean().reset_index()
        return result_df

    def process_series(self, start, end):
        """
        Process a range of series IDs and save to storage.

        Args:
            start (int): Start index of series IDs.
            end (int): End index of series IDs.
        """
        try:
            series_df = pd.read_csv(self.series_file)
            logger.info(f"Loaded {self.series_file} with columns: {series_df.columns.tolist()}")
            if "series_id" not in series_df.columns:
                raise ValueError("Missing 'series_id' column in series file")
            series_ids = series_df["series_id"].tolist()
            series_ids = series_ids[start:end]
        except Exception as e:
            logger.error(f"Error reading series IDs: {e}")
            raise

        for i in range(0, len(series_ids), self.batch_size):
            batch_ids = series_ids[i:i + self.batch_size]
            result_df = self.fetch_batch(batch_ids)
            if result_df is not None:
                try:
                    if IS_DATABRICKS:
                        spark_df = self.spark.createDataFrame(result_df)
                        spark_df = spark_df.withColumn("series_id", F.lit("_".join(batch_ids)))
                        spark_df.write.format("delta").mode("overwrite").partitionBy("series_id").save(f"{self.output_path}/data")
                        self.fs.write_table(
                            name="fred_economic_indicators",
                            df=spark_df,
                            description="FRED economic indicators for stock prediction",
                            primary_keys=["date", "series_id"]
                        )
                        logger.info(f"Saved batch {batch_ids} to Delta Lake at {self.output_path}/data")
                    else:
                        for series_id in batch_ids:
                            if series_id in result_df.columns:
                                df = result_df[["date", series_id]].dropna()
                                output_file = Path(self.output_path) / f"{series_id}.csv"
                                df.to_csv(output_file, index=False)
                                logger.info(f"Saved {series_id} to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving batch {batch_ids}: {e}")
                    continue

    def run_etl(self):
        """Run the full ETL pipeline."""
        try:
            series_df = pd.read_csv(self.series_file)
            total_series = len(series_df)
            logger.info(f"Processing {total_series} series")
        except Exception as e:
            logger.error(f"Error loading {self.series_file}: {e}")
            raise

        chunk_size = 100
        for start in range(0, total_series, chunk_size):
            end = min(start + chunk_size, total_series)
            self.process_series(start, end)
        logger.info(f"ETL completed, data saved to {self.output_path}")

if __name__ == "__main__":
    API_KEY = "17188a6953269ab608ba14c3e3d8fb02"  # Your FRED API key
    SERIES_FILE = "src/data/indicators.csv"
    OUTPUT_PATH = "src/data/fred_economic_indicators" if not IS_DATABRICKS else "/mnt/fred_data"
    fetcher = FREDDataFetcher(api_key=API_KEY, series_file=SERIES_FILE, output_path=OUTPUT_PATH, batch_size=5)
    fetcher.run_etl()