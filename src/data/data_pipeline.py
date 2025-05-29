import pandas as pd
import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa
from typing import List, Optional

# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('etl/data_pipeline.txt'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

def load_fred_data(fred_dir: str, is_databricks: bool = False, delta_path: str = None) -> pd.DataFrame:
    """Load and combine FRED data from staged CSV files or Delta Lake."""
    try:
        if is_databricks:
            spark = SparkSession.builder.appName("FREDDataLoader").getOrCreate()
            spark_df = spark.read.format("delta").load(delta_path)
            df = spark_df.toPandas()
            logger.info(f"Loaded FRED data from Delta Lake at {delta_path}")
        else:
            fred_dir = Path(fred_dir)
            dfs = []
            for file in fred_dir.glob("*.csv"):
                series_id = file.stem
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df[[series_id]]  # Keep only the series column
                dfs.append(df)
            if not dfs:
                raise ValueError(f"No CSV files found in {fred_dir}")
            df = pd.concat(dfs, axis=1, join="outer")
            df = df.fillna(method='ffill').fillna(method='bfill')  # Fill missing values
            logger.info(f"Loaded {len(dfs)} FRED series from {fred_dir}")
        return df
    except Exception as e:
        logger.error(f"Error loading FRED data: {str(e)}")
        raise

def load_yfinance_data(ticker: str, yf_dir: str) -> pd.DataFrame:
    """Load Yahoo Finance data for a single ticker from staged CSV."""
    try:
        file_path = Path(yf_dir) / f"{ticker.upper()}_monthly.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {ticker} at {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Close']].rename(columns={'Close': f'{ticker}_Close'})
        logger.info(f"Loaded Yahoo Finance data for {ticker} from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Yahoo Finance data for {ticker}: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame, target_col: str, lag_periods: List[int] = [1, 3, 6], ma_windows: List[int] = [3, 12]) -> pd.DataFrame:
    """Add lagged variables and moving averages for features."""
    try:
        logger.info("Engineering features: lags and moving averages")
        df = df.copy()
        
        # Create lagged features for all columns
        for col in df.columns:
            for lag in lag_periods:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Create moving averages for target column (stock price)
        for window in ma_windows:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window).mean()
        
        # Drop rows with NaN values from lags or moving averages
        df = df.fillna(method='bfill').fillna(method='ffill')
        logger.info(f"Feature engineering complete, resulting shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def save_to_storage(df: pd.DataFrame, output_path: str, ticker: str, is_databricks: bool = False) -> None:
    """Save DataFrame to Parquet or Delta Lake."""
    try:
        output_file = f"{output_path}/{ticker}_combined.csv"
        df.to_csv(output_file , index=True)  # Save as CSV first for logging
        output_file = f"{output_path}/{ticker}_combined.parquet"
        
        logger.info(f"Saving combined data for {ticker} to {output_file}")
        
        if is_databricks:
            spark = SparkSession.builder.appName("CombinedDataSaver").getOrCreate()
            spark_df = spark.createDataFrame(df.reset_index())
            spark_df.write.format("delta").mode("overwrite").save(output_file)
        else:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_file)
        
        logger.info(f"Data saved successfully for {ticker} to {output_file}")
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {str(e)}")
        raise

def main(
    fred_dir: str,
    yf_dir: str,
    tickers: List[str],
    output_path: str,
    lag_periods: List[int] = [1, 3, 6],
    ma_windows: List[int] = [3, 12],
    delta_path: Optional[str] = None
) -> None:
    """Join FRED and Yahoo Finance data for each ticker, engineer features, and save."""
    try:
        # Load FRED data
        fred_data = load_fred_data(fred_dir, IS_DATABRICKS, delta_path)
        
        # Process each ticker
        for ticker in tickers:
            try:
                logger.info(f"Processing ticker: {ticker}")
                # Load Yahoo Finance data
                stock_data = load_yfinance_data(ticker, yf_dir)
                
                # Join FRED and stock data
                combined_data = fred_data.join(stock_data, how='inner').dropna()
                if combined_data.empty:
                    logger.warning(f"Empty dataset after joining for {ticker}")
                    continue
                
                # Engineer features
                combined_data = engineer_features(combined_data, f'{ticker}_Close', lag_periods, ma_windows)
                
                # Save to storage
                save_to_storage(combined_data, output_path, ticker, IS_DATABRICKS)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        logger.info("ETL pipeline completed successfully")
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    import pandas as pd
    stocks = pd.read_csv('C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/heavy-stocks.csv')
    stocks = stocks['Symbol'].dropna().unique().tolist()
    parser = argparse.ArgumentParser(description="Stock Price Forecasting ETL Pipeline")
    parser.add_argument('--fred-dir', type=str, default='C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/fred_economic_indicators3', help="Directory with FRED CSV files")
    parser.add_argument('--yf-dir', type=str, default='C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/yf_monthly', help="Directory with Yahoo Finance CSV files")
    
    parser.add_argument('--tickers', type=str, nargs='+', default=stocks, help="List of stock tickers")
    parser.add_argument('--output-path', type=str, default='C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/combined', help="Output directory for combined Parquet files")
    parser.add_argument('--delta-path', type=str, default='/mnt/fred_data', help="Delta Lake path for Databricks")
    
    args = parser.parse_args()
    
    main(
        fred_dir=args.fred_dir,
        yf_dir=args.yf_dir,
        tickers=args.tickers,
        output_path=args.output_path,
        delta_path=args.delta_path
    )