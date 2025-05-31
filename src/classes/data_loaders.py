import pandas as pd
from pathlib import Path
import yfinance as yf
import requests
import logging
from dotenv import load_dotenv, find_dotenv
import os

# This will search parent directories for .env
load_dotenv(find_dotenv())

# Now you can access your key
api_key = os.getenv("FRED_API")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_stock_data(ticker: str, start_date: str, end_date: str, data_dir: str) -> pd.DataFrame:
    """Fetch monthly adjusted stock data from yfinance."""
    output_dir = Path(data_dir) / "yf_monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Fetching monthly data for {ticker}")
    try:
        data = pd.read_csv(output_dir / f"{ticker}_monthly.csv")
        data['Date'] = pd.to_datetime(data['Date'])
    except FileNotFoundError:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data[f'{ticker}_Close'] = data['Close'].round(4)
        data[f'{ticker}_Volume'] = data['Volume']
        data = data[['Date', f'{ticker}_Close', f'{ticker}_Volume']]
        data.to_csv(output_dir / f"{ticker}_monthly.csv", index=False)
    
    data['Date'] = data['Date'].dt.to_period('M').dt.to_timestamp()
    data = data.set_index('Date').loc[~data.index.duplicated(keep='first')]
    return data.dropna()

def get_indicator_data(series_id: str, start_date: str, end_date: str, api_key: str, data_dir: str) -> pd.DataFrame:
    """Fetch FRED indicator data."""
    output_dir = Path(data_dir) / "fred_economic_indicators"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Fetching FRED data for {series_id}")
    try:
        df = pd.read_csv(output_dir / f"{series_id}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "realtime_start": start_date,
            "realtime_end": end_date,
            "observation_start": start_date,
            "observation_end": end_date,
            "file_type": "json"
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if "observations" not in data or not data["observations"]:
                logger.warning(f"No observations for {series_id}")
                return pd.DataFrame()
            df = pd.DataFrame(data["observations"])[["date", "value"]]
            df["Date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["Date", "value"]]
            df = df.rename(columns={"value": series_id})
            df.to_csv(output_dir / f"{series_id}.csv", index=False)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()
    
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').loc[~df.index.duplicated(keep='first')]
    return df.dropna()