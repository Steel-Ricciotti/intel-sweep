import pandas as pd
from pathlib import Path
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for generating financial features like lags and moving averages."""
    
    def __init__(self, lag_periods: List[int] = [1, 3, 6], ma_windows: List[int] = [3, 12], data_dir: str = None):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            lag_periods (List[int]): Periods for lagged features.
            ma_windows (List[int]): Windows for moving averages.
            data_dir (str, optional): Directory for saving features.
        """
        self.lag_periods = lag_periods
        self.ma_windows = ma_windows
        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir:
            self.features_dir = self.data_dir / "features"
            self.features_dir.mkdir(parents=True, exist_ok=True)

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Generate lagged variables, moving averages, and volatility features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with Date index.
            target_col (str): Target column for feature engineering (e.g., 'BTC-USD_Close').
        
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        logger.info("Engineering features: lags and moving averages")
        df = df.copy()
        print(df.head())
        ticker = target_col.split('_')[0]
        
        df.index = pd.to_datetime(df.index)
        
        # Lagged features for indicators and volume
        for col in df.columns:
            if col != target_col:
                for lag in self.lag_periods:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        print(df.head())
        # Moving averages and volatility for target
        for window in self.ma_windows:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_volatility'] = df[target_col].rolling(window=3).std()
        
        # Drop NaNs and ensure unique index
        df = df.dropna()
        df = df.reset_index().drop_duplicates(subset='Date', keep='first').set_index('Date')
        
        # Save features if data_dir is provided
        if self.data_dir:
            output_path = self.features_dir / f"{ticker}_combined.csv"
            logger.info(f"Saving features to {output_path}")
            df.to_csv(output_path, index=True)
        
        logger.info(f"Engineered features: {df.columns.tolist()}")
        return df