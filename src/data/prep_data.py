import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(stock_ticker='AAPL', fred_files=None, stock_file=None, lags=3, test_size=0.2):
    """
    Merge stock and FRED data, create lagged features, and split into train/test sets.
    
    Args:
        stock_ticker (str): Stock ticker (e.g., 'AAPL').
        fred_files (list): List of FRED CSV files.
        stock_file (str): Path to stock CSV file.
        lags (int): Number of lag periods for features.
        test_size (float): Proportion of data for test set.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Load stock data
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df[['Date', 'Close', 'Volume']].set_index('Date')

    # Load and merge FRED data
    fred_dfs = []
    for fred_file in fred_files:
        df = pd.read_csv(fred_file)
        df['date'] = pd.to_datetime(df['date'])
        series_id = fred_file.stem  # e.g., '00XALCATM086NEST'
        df = df[['date', series_id]].set_index('date')
        fred_dfs.append(df)
    fred_df = pd.concat(fred_dfs, axis=1)

    # Merge stock and FRED data
    fred_df.reset_index(inplace=True)
    fred_df['date'] = pd.to_datetime(fred_df['date']).dt.to_period('M').dt.to_timestamp()
    fred_df.set_index('date',inplace=True)
    stock_df.reset_index(inplace=True)
    stock_df.drop('Volume', axis=1, inplace=True)
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.to_period('M').dt.to_timestamp()
    stock_df.rename(columns={'Date': 'date'}, inplace=True)
    stock_df.set_index('date', inplace=True)
    data = stock_df.join(fred_df, how='inner').dropna()

    # Create lagged features
    features = []
    feature_names = []
    for col in data.columns:
        for lag in range(1, lags + 1):
            data[f'{col}_lag{lag}'] = data[col].shift(lag)
            feature_names.append(f'{col}_lag{lag}')
    data['target'] = data['Close'].shift(-2)  # Next month's Close price
    data = data.dropna()

    # Split train/test
    train_size = int(len(data) * (1 - test_size))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    train.to_csv('src/data/train/train_data.csv', index=True)
    test.to_csv('src/data/test/test_data.csv', index=True)

    X_train = train[feature_names]
    y_train = train['target']
    X_test = test[feature_names]
    y_test = test['target']

    logger.info(f"Prepared data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples, {len(feature_names)} features")
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    fred_files = [
        Path("data/fred_economic_indicators/00XALCATM086NEST.csv"),
        Path("data/fred_economic_indicators/00XALCBEM086NEST.csv"),
        Path("data/fred_economic_indicators/00XALCCZM086NEST.csv")
    ]
    stock_file = Path("data/av_monthly/AAPL_monthly.csv")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(fred_files=fred_files, stock_file=stock_file)
    print(f"Features: {feature_names}")
