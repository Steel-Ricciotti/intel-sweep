# ISM New Orders vs Industrials/Logistics
leading_indicators = 'UMDMNO'
# Manufacturers' New Orders: Durable Goods 
equities = ['CAT','EMR','FDX','UPS']

# Money Supply




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import logging
from pathlib import Path
import yfinance as yf
import requests
from dash import Dash, html, dcc, dash_table
import plotly.express as px
from typing import List

#impport FRED_API_KEY from .env
from dotenv import load_dotenv, find_dotenv
import os

# This will search parent directories for .env
load_dotenv(find_dotenv())

# Now you can access your key
api_key = os.getenv("FRED_API")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETL:


    def get_stock_data(ticker, start_date="2010-01-01", end_date="2025-04-30"):
        """Fetch monthly adjusted stock data from yfinance."""
        output_dir = Path("C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/yf_monthly")
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

    def get_indicator_data(series_id, start_date="2010-01-01", end_date="2025-04-30", api_key="17188a6953269ab608ba14c3e3d8fb02"):
        """Fetch FRED indicator data."""
        output_dir = Path("C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/fred_economic_indicators")
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
                    return None
                df = pd.DataFrame(data["observations"])[["date", "value"]]
                df["Date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df[["Date", "value"]]
                df = df.rename(columns={"value": series_id})
                df.to_csv(output_dir / f"{series_id}.csv", index=False)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {series_id}: {e}")
                return None
        
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        df = df.set_index('Date').loc[~df.index.duplicated(keep='first')]
        return df.dropna()

    def merge_indicators_with_stock_data(stock_list, indicators, start_date="2010-01-01", end_date="2025-04-30"):
        """Merge stock data with economic indicators."""
        logger.info("Merging stock data with economic indicators")
        merged_data = get_stock_data(stock_list[0], start_date, end_date)
        for stock in stock_list[1:]:
            stock_data = get_stock_data(stock, start_date, end_date)
            merged_data = merged_data.merge(stock_data, left_index=True, right_index=True, how='inner')

        for indicator in indicators:
            indicator_data = get_indicator_data(indicator, start_date, end_date)
            if indicator_data is not None:
                merged_data = merged_data.merge(indicator_data, left_index=True, right_index=True, how='inner')
                logger.info(f"Added indicator {indicator} to stock data")
            else:
                logger.warning(f"Indicator {indicator} data not available")

        merged_data = merged_data.loc[start_date:end_date]
        merged_data = merged_data.loc[~merged_data.index.duplicated(keep='first')]
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill').dropna()
        output_dir = Path("C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/combined")
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_dir / "m2-bitcoin-gold_combined.csv", index=True)
        return merged_data

    def engineer_features(df: pd.DataFrame, target_col: str, lag_periods: List[int] = [1, 3, 6], ma_windows: List[int] = [3, 12]) -> pd.DataFrame:
        """Add lagged variables and moving averages for features."""
        logger.info("Engineering features: lags and moving averages")
        df = df.copy()
        ticker = target_col.split('_')[0]
        
        df.index = pd.to_datetime(df.index)
        
        # Lagged features for indicators and volume
        for col in df.columns:
            if col != target_col:  # Lag all columns except target
                for lag in lag_periods:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Moving averages and volatility for target
        for window in ma_windows:
            df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_volatility'] = df[target_col].rolling(window=3).std()
        
        # Drop NaNs and ensure unique index
        df = df.dropna()
        df = df.reset_index().drop_duplicates(subset='Date', keep='first').set_index('Date')
        output_dir = Path("C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{ticker}_combined.csv", index=True)
        logger.info(f"Engineered features: {df.columns.tolist()}")
        return df

class StockPriceLSTM(nn.Module):
    """LSTM network for stock price forecasting."""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_pytorch_forecast(data, target, selected_features, horizon=1, params=None, use_lstm=True):
    """Train PyTorch model for h-month-ahead forecasting."""
    logger.info(f"Starting training for {target}, horizon {horizon}")
    if params is None:
        params = {'hidden_size': 128, 'epochs': 200, 'learning_rate': 0.0005, 'batch_size': 32}

    try:
        df = data.copy()
        logger.info(f"Loaded data for {target} with shape {df.shape}")
        if target not in df.columns:
            logger.error(f"Target column {target} not found in data.")
            return None, None, None

        # Validate features
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return None, None, None

        df[target] = pd.to_numeric(df[target], errors='coerce')
        if df[target].isna().any():
            logger.warning(f"Target {target} contains {df[target].isna().sum()} NaN values.")
            df = df.dropna(subset=[target])
            if df.empty:
                logger.error(f"No valid data for {target} after numeric conversion.")
                return None, None, None

        df[target + f'_t{horizon}'] = df[target].shift(-horizon)
        df = df.dropna(subset=selected_features + [target + f'_t{horizon}'])
        logger.info(f"Data shape after NaN handling: {df.shape}")
        if df.empty:
            logger.error(f"No valid data for {target} after NaN handling.")
            return None, None, None

        X = df[selected_features]
        y = df[target + f'_t{horizon}']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Scale features and target
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_test.values

        if use_lstm:
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

        input_size = X.shape[1]
        model = StockPriceLSTM(input_size, params['hidden_size'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        mlflow.set_experiment("stock_price_forecasting")
        with mlflow.start_run(run_name=f"{target}_horizon{horizon}"):
            mlflow.set_tag("ticker", target.split('_')[0])
            mlflow.set_tag("horizon", horizon)
            mlflow.set_tag("model_type", "LSTM")
            model.train()
            for epoch in range(params['epochs']):
                for i in range(0, len(X_train_tensor), params['batch_size']):
                    batch_X = X_train_tensor[i:i + params['batch_size']]
                    batch_y = y_train_tensor[i:i + params['batch_size']]
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{params['epochs']} - Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_tensor).numpy().flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            metrics = calculate_metrics(y_test, y_pred)
            logger.info(f"Metrics for {target} (horizon {horizon}): {metrics}")

            current_prices = df.loc[y_test.index, target]
            if len(y_pred) != len(current_prices):
                logger.error(f"Shape mismatch: y_pred ({len(y_pred)}), current_prices ({len(current_prices)})")
                return None, None, None

            price_change = (y_pred - current_prices) / current_prices * 100
            signals = np.where(price_change > 5, "Buy", np.where(price_change < -5, "Sell", "Hold"))
            backtest_df = pd.DataFrame({
                "Date": y_test.index,
                "Actual": y_test,
                "Predicted": y_pred,
                "Current_Price": current_prices,
                "Signal": signals
            })
            backtest_df.to_csv(f"backtest_{target}_horizon{horizon}.csv", index=False)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("features", ", ".join(selected_features))
            mlflow.log_artifact(f"backtest_{target}_horizon{horizon}.csv")
            mlflow.pytorch.log_model(model, "pytorch_model", input_example=X_train_scaled[:1])
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/pytorch_model"
            registered_model = mlflow.register_model(model_uri, f"StockPricePyTorch_Horizon{horizon}")
            logger.info(f"Model registered: {registered_model.name} version {registered_model.version}")

        return model, metrics, backtest_df
    except Exception as e:
        logger.error(f"Error in train_pytorch_forecast for {target} horizon {horizon}: {str(e)}")
        return None, None, None
