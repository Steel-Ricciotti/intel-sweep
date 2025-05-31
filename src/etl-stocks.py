import os
import time
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YFinanceStockFetcher:
    def __init__(self, tickers, output_dir="src/data/yf_monthly"):
        """
        Args:
            tickers (list[str]): List of ticker symbols.
            output_dir (str): Directory to save CSV files.
        """
        self.tickers = tickers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def fetch_monthly_data(self, ticker):
        """
        Fetch monthly adjusted stock data from yfinance.
        """
        logger.info(f"Fetching monthly data for {ticker}")
        # Use yfinance download with monthly interval
        data = yf.download(ticker, period="max", interval="1mo", auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Reset and rename columns for consistency with Alpha Vantage
        data.reset_index(inplace=True)
        data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)
        # Add 'Adj Close' (already adjusted due to auto_adjust=True)
        data['Adj Close'] = data['Close']

        # Keep relevant columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        data = data.sort_values('Date')
        data = data.round(4)
        return data

    def run(self):
        failed = []
        for ticker in self.tickers:
            try:
                df = self.fetch_monthly_data(ticker)
                output_file = self.output_dir / f"{ticker.upper()}_monthly.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved data to {output_file}")
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                failed.append(ticker)
            time.sleep(1)  # Avoid yfinance rate-limiting
        if failed:
            logger.warning(f"Failed tickers: {failed}")
            return failed
        return []



if __name__ == "__main__":
    # Path to your CSV file with S&P 500 tickers

    csv_path = "C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/equities.csv"
    df_symbols = pd.read_csv(csv_path)
    symbols = df_symbols['Symbol'].dropna().unique().tolist()
    fetcher = YFinanceStockFetcher(tickers=symbols)
    failed_tickers = fetcher.run()