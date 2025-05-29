import pandas as pd
import yfinance as yf
import logging
import os
import time
from pathlib import Path
from retry import retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance and saves as CSV files."""

    def __init__(self, tickers, output_dir="data/stocks", start_date="2015-03-15", end_date="2017-05-15"):
        """
        Initialize the stock data fetcher.

        Args:
            tickers (list): List of stock ticker symbols.
            output_dir (str): Directory to save CSV files.
            start_date (str): Start date for data (YYYY-MM-DD).
            end_date (str): End date for data (YYYY-MM-DD).
        """
        self.tickers = tickers
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date
        os.makedirs(self.output_dir, exist_ok=True)

    @retry(tries=3, delay=1, backoff=2, exceptions=(Exception,))
    def fetch_stock(self, ticker):
        """
        Fetch data for a single stock ticker.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with stock data, or None if failed.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date, interval="1d")
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            # Reset index to make 'Date' a column
            df = stock.history()['Close']
            # Select and rename columns to match original format
            df = df[['Date', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')
            df = df.round(4)
            logger.info(f"Fetched data for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def run(self):
        """Fetch and save data for all tickers as individual CSV files."""
        failed_tickers = []
        for ticker in self.tickers:
            df = self.fetch_stock(ticker)
            if df is not None:
                output_file = os.path.join(self.output_dir, f"{ticker.lower()}.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {ticker} to {output_file}")
            else:
                failed_tickers.append(ticker)
            time.sleep(0.5)  # Rate limiting
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers}")


if __name__ == "__main__":
    # Subset of tickers for testing (some may be delisted)
    TICKERS = ['MSFT']
    fetcher = StockDataFetcher(TICKERS)
    fetcher.run()