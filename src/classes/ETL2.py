import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional
from src.classes.data_loaders import get_stock_data, get_indicator_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv, find_dotenv
import os

# This will search parent directories for .env
load_dotenv(find_dotenv())

# Now you can access your key
api_key = os.getenv("FRED_API")

class ETL:
    """Extract, Transform, Load class for financial data processing."""
    
    def __init__(
        self,
        data_dir: str,
        start_date: str = "2010-01-01",
        end_date: str = "2025-04-30",
        api_key: str = api_key
    ):
        """
        Initialize ETL with configuration parameters.
        
        Args:
            data_dir (str): Root directory for data storage.
            start_date (str): Start date for data fetching.
            end_date (str): End date for data fetching.
            api_key (str): FRED API key for indicator data.
        """
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.combined_dir = self.data_dir / "combined"
        self.features_dir = self.data_dir / "features"
        self.combined_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def extract_stock(self, ticker: str) -> pd.DataFrame:
        """
        Extract stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'BTC-USD').
        
        Returns:
            pd.DataFrame: Stock data with Date index.
        """
        logger.info(f"Extracting stock data for {ticker}")
        try:
            data = get_stock_data(ticker, self.start_date, self.end_date, self.data_dir)
            return data
        except Exception as e:
            logger.error(f"Failed to extract stock data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def extract_indicator(self, series_id: str) -> Optional[pd.DataFrame]:
        """
        Extract economic indicator data from FRED.
        
        Args:
            series_id (str): FRED series ID (e.g., 'M2SL').
        
        Returns:
            Optional[pd.DataFrame]: Indicator data with Date index, or None if failed.
        """
        logger.info(f"Extracting indicator data for {series_id}")
        try:
            data = get_indicator_data(series_id, self.start_date, self.end_date, self.api_key, self.data_dir)
            return data
        except Exception as e:
            logger.error(f"Failed to extract indicator data for {series_id}: {str(e)}")
            return None

    def transform_merge(self, stock_list: List[str], indicators: List[str], experiment_name: str) -> pd.DataFrame:
        """
        Merge stock and indicator data, ensuring clean and aligned output.
        
        Args:
            stock_list (List[str]): List of stock tickers (e.g., ['BTC-USD', 'GLD']).
            indicators (List[str]): List of FRED series IDs (e.g., ['M2SL']).
            experiment_name (str): Name for the experiment (used in output file naming).
        
        Returns:
            pd.DataFrame: Merged and cleaned DataFrame with Date index.
        """
        logger.info(f"Merging data for experiment: {experiment_name}")
        if not stock_list:
            logger.error("No stocks provided for merging")
            return pd.DataFrame()

        # Extract and merge stock data
        merged_data = self.extract_stock(stock_list[0])
        # if merged_data.empty:
        #     logger.error(f"Initial stock data empty for {stock_list[0]}")
        #     return pd.DataFrame()

        # for stock in stock_list:
        #     stock_data = self.extract_stock(stock)
        #     if not stock_data.empty:
        #         merged_data = merged_data.merge(stock_data, left_index=True, right_index=True, how='inner')
        #         #merged_data = merged_data.join(stock_data)
        #         logger.info(f"Merged stock data for {stock}")
        #     else:
        #         logger.warning(f"No data for stock {stock}, skipping")

        # Merge indicator data
        for indicator in indicators:
            indicator_data = self.extract_indicator(indicator)
            if indicator_data is not None:
                merged_data = merged_data.merge(indicator_data, left_index=True, right_index=True, how='inner')
                logger.info(f"Added indicator {indicator} to merged data")
            else:
                logger.warning(f"Indicator {indicator} data not available, skipping")

        # Clean and filter data
        merged_data = merged_data.loc[self.start_date:self.end_date]
        merged_data = merged_data.loc[~merged_data.index.duplicated(keep='first')]
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill').dropna()
        
        if merged_data.empty:
            logger.error(f"No data after merging for {experiment_name}")
            return pd.DataFrame()

        return merged_data

    def load(self, data: pd.DataFrame, filename: str, save_features: bool = False) -> None:
        """
        Save processed data to CSV.
        
        Args:
            data (pd.DataFrame): Data to save.
            filename (str): Output filename (e.g., 'm2_equities_combined.csv').
            save_features (bool): If True, save to features directory; else, combined directory.
        """
        if data.empty:
            logger.error("No data to save")
            return

        output_dir = self.features_dir if save_features else self.combined_dir
        output_path = output_dir / filename
        logger.info(f"Saving data to {output_path}")
        try:
            data.to_csv(output_path, index=True)
        except Exception as e:
            logger.error(f"Failed to save data to {output_path}: {str(e)}")

    def run(self, stock_list: List[str], indicators: List[str], experiment_name: str) -> pd.DataFrame:
        """
        Execute full ETL pipeline for a given experiment.
        
        Args:
            stock_list (List[str]): List of stock tickers.
            indicators (List[str]): List of FRED series IDs.
            experiment_name (str): Name for the experiment.
        
        Returns:
            pd.DataFrame: Merged and cleaned data.
        """
        logger.info(f"Running ETL pipeline for experiment: {experiment_name}")
        merged_data = self.transform_merge(stock_list, indicators, experiment_name)
        if not merged_data.empty:
            self.load(merged_data, f"{experiment_name}_combined.csv")
        return merged_data