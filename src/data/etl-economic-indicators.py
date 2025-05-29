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
    import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FREDDataFetcher:
    """Fetches and processes FRED economic data for storage."""

    def __init__(self, api_key, series_file, output_path="data/processed/fred_economic_indicators", batch_size=1):
        """
        Initialize the FRED data fetcher.

        Args:
            api_key (str): FRED API key.
            series_file (str): Path to CSV file containing series IDs.
            output_path (str): Path for storing processed data (Parquet locally, Delta Lake in Databricks).
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
        else:
            self.spark = None
            self.fs = None

    @retry(tries=3, delay=1, backoff=2, exceptions=(requests.exceptions.RequestException,))
    def fetch_series(self, series_id, start_date="2015-01-01"):
        """
        Fetch FRED data for a single series synchronously.

        Args:
            series_id (str): FRED series ID.
            start_date (str): Start date for observations (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with date and value columns.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "realtime_start": start_date,
            "observation_start": start_date,
            "frequency": "m",
            "file_type": "json"
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data["observations"])[["date", "value"]]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": series_id})
            logger.info(f"Fetched data for {series_id}")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {series_id}: {e}")
            raise

    def fetch_batch(self, series_ids, start_date="2000-01-01"):
        """
        Fetch data for a batch of series IDs synchronously.

        Args:
            series_ids (list): List of FRED series IDs.
            start_date (str): Start date for observations.

        Returns:
            pd.DataFrame: Consolidated DataFrame for the batch.
        """
        result_df = None
        for series_id in series_ids:
            try:
                df = self.fetch_series(series_id, start_date)
                if result_df is None:
                    result_df = df
                else:
                    result_df = result_df.merge(df, on="date", how="outer")
            except Exception as e:
                logger.error(f"Skipping {series_id} due to error: {e}")
                continue
            time.sleep(0.5)  # Rate limiting: ~2 requests/second
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
            if "File" not in series_df.columns:
                raise ValueError("Missing 'File' column in Months.csv")
            series_ids = series_df["File"].tolist()
            # Clean series IDs: remove '0\' prefix and '.csv' suffix
            series_ids = [item.split("\\")[-1].split(".csv")[0] for item in series_ids]
            logger.info(f"Extracted series IDs: {series_ids}")
            series_ids = series_ids[start:end]
        except Exception as e:
            logger.error(f"Error reading series IDs: {e}")
            raise

        for i in range(0, len(series_ids), self.batch_size):
            batch_ids = series_ids[i:i + self.batch_size]
            result_df = self.fetch_batch(batch_ids)
            if result_df is not None:
                result_df = result_df.dropna(how="all", subset=result_df.columns[1:])
                df = result_df.set_index("date", inplace=True)
                df_monthly = df.resample("MS").mean()
                # Optional: reset index if needed
                result_df = df_monthly.reset_index(inplace=True)
                output_file = f"{self.output_path}/batch_{start + i}_{start + i + len(batch_ids)}"

                try:
                    if IS_DATABRICKS:
                        spark_df = self.spark.createDataFrame(result_df)
                        spark_df.write.format("delta").mode("append").save(self.output_path)
                        self.fs.write_table(
                            name="fred_economic_indicators",
                            df=spark_df,
                            description="FRED economic indicators for stock prediction",
                            primary_keys=["date"]
                        )
                        logger.info(
                            f"Saved batch {start + i}-{start + i + len(batch_ids)} to Delta Lake at {self.output_path}")
                    else:
                        result_df.to_csv(f"{output_file}.csv")
                        logger.info(
                            f"Saved batch {start + i}-{start + i + len(batch_ids)} to Parquet at {output_file}.parquet")
                except Exception as e:
                    logger.error(f"Error saving batch {start + i}-{start + i + len(batch_ids)}: {e}")
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

        chunk_size = 1000
        for start in range(0, total_series, chunk_size):
            end = min(start + chunk_size, total_series)
            self.process_series(start, end)
        logger.info(f"ETL completed, data saved to {self.output_path}")


if __name__ == "__main__":
    API_KEY = "17188a6953269ab608ba14c3e3d8fb02"  # Your FRED API key
    SERIES_FILE = "Months.csv"
    #OUTPUT_PATH = "data/processed/fred_economic_indicators" if not IS_DATABRICKS else "/mnt/fred_data"
    OUTPUT_PATH = "data/processed/fred_economic_indicators" if not IS_DATABRICKS else "/mnt/fred_data"
    fetcher = FREDDataFetcher(API_KEY, SERIES_FILE, OUTPUT_PATH)
    fetcher.run_etl()