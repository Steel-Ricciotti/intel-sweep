import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import logging
from pathlib import Path
from dash import Dash, html, dcc, dash_table
import plotly.express as px
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

class StockPriceNN(nn.Module):
    """Feed-forward neural network for stock price forecasting."""
    def __init__(self, input_size, hidden_size=64):
        super(StockPriceNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StockPriceLSTM(nn.Module):
    """LSTM network for stock price forecasting."""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
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

def train_pytorch_forecast(data_path, target, selected_features, horizon=1, params=None, use_lstm=False):
    """Train PyTorch model for h-month-ahead forecasting."""
    print("Test: Entering train_pytorch_forecast")
    if params is None:
        params = {'hidden_size': 64, 'epochs': 100, 'learning_rate': 0.001, 'batch_size': 16}

    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data for {target} with shape {df.shape}")
        if target not in df.columns:
            logger.error(f"Target column {target} not found in data.")
            return None, None, None

        df[target] = pd.to_numeric(df[target], errors='coerce')
        if df[target].isna().any():
            logger.warning(f"Target {target} contains {df[target].isna().sum()} NaN values after conversion.")
            df = df.dropna(subset=[target])
            if df.empty:
                logger.error(f"No valid data for {target} after numeric conversion.")
                return None, None, None
        logger.info(f"Target {target} dtype: {df[target].dtype}")

        if len(df) < horizon:
            logger.error(f"Dataset too short ({len(df)} rows) for horizon {horizon}.")
            return None, None, None

        df[target + f'_t{horizon}'] = df[target].shift(-horizon)
        df = df.dropna(subset=selected_features + [target + f'_t{horizon}'])
        logger.info(f"Data shape after NaN handling: {df.shape}")
        if df.empty:
            logger.error(f"No valid data for {target} after NaN handling.")
            return None, None, None

        X = df[selected_features].fillna(method="ffill").fillna(method="bfill")
        y = df[target + f'_t{horizon}']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        logger.info(f"Train indices: {X_train.index[:5].tolist()}, Test indices: {X_test.index[:5].tolist()}")
        y_train = pd.to_numeric(y_train, errors='coerce')
        y_test = pd.to_numeric(y_test, errors='coerce')
        non_nan_train = ~y_train.isna()
        non_nan_test = ~y_test.isna()
        X_train = X_train[non_nan_train]
        y_train = y_train[non_nan_train]
        X_test = X_test[non_nan_test]
        y_test = y_test[non_nan_test]
        logger.info(f"Train size after NaN filter: {len(X_train)}, Test size after NaN filter: {len(X_test)}")
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error(f"No valid train/test data for {target} after NaN handling.")
            return None, None, None

        print("Test2: Before scaling")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if use_lstm:
            # Reshape for LSTM [samples, timesteps, features]
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

        input_size = X.shape[1]
        model = StockPriceLSTM(input_size, params['hidden_size']) if use_lstm else StockPriceNN(input_size, params['hidden_size'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        mlflow.set_experiment("stock_price_forecasting")
        with mlflow.start_run(run_name=f"{target}_horizon{horizon}"):
            mlflow.set_tag("ticker", target.split('_')[0])
            mlflow.set_tag("horizon", horizon)
            mlflow.set_tag("model_type", "LSTM" if use_lstm else "NN")
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
                y_pred = model(X_test_tensor).numpy().flatten()
            # Bias correction
            with torch.no_grad():
                y_train_pred = model(X_train_tensor).numpy().flatten()
            bias = np.mean(y_train_pred - y_train)
            y_pred += bias
            metrics = calculate_metrics(y_test, y_pred)
            logger.info(f"Metrics for {target} (horizon {horizon}): {metrics}")

            current_prices = df.loc[y_test.index, target]
            logger.info(f"Current prices shape: {current_prices.shape}, y_test shape: {y_test.shape}")
            price_change = (y_pred - current_prices) / current_prices * 100
            signals = np.where(price_change > 5, "Buy", np.where(price_change < -5, "Sell", "Hold"))
            backtest_df = pd.DataFrame({
                "Date": y_test.index,
                "Actual": y_test,
                "Predicted": y_pred,
                "Current_Price": current_prices,
                "Signal": signals
            })
            backtest_df.to_csv(f"backtest_{target}_horizon{horizon}.csv")
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
        print(f"Error in train_pytorch_forecast for {target} horizon {horizon}: {str(e)}")
        logger.error(f"Error in train_pytorch_forecast for {target} horizon {horizon}: {str(e)}")
        return None, None, None

def getStockFeatures(ticker="AAPL"):
    print("Test: Entering getStockFeatures")
    """Get top 10 features for a stock using Random Forest."""
    data_path = Path(f"C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/combined/{ticker}_combined.parquet")
    if not data_path.exists():
        logger.error(f"Data file for {ticker} not found at {data_path}.")
        return []
    df = pd.read_parquet(data_path)
    target = f"{ticker}_Close"
    if target not in df.columns:
        logger.error(f"Target column {target} not found in data for {ticker}.")
        return []
    df[target] = pd.to_numeric(df[target], errors='coerce')
    if df[target].isna().any():
        logger.warning(f"Target {target} contains {df[target].isna().sum()} NaN values after conversion.")
        df = df.dropna(subset=[target])
    features = [col for col in df.columns if col != target and not col.startswith(target + '_t')]
    logger.info(f"Found {len(features)} features for {ticker}: {features}")
    if not features:
        logger.error(f"No valid features found for {ticker}.")
        return []

    df = df.dropna(subset=features + [target])
    logger.info(f"Data shape after NaN handling for {ticker}: {df.shape}")
    if df.empty:
        logger.error(f"No valid data for {ticker} after NaN handling.")
        return []

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring="r2")
    logger.info(f"Cross-validated R-squared for {ticker}: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    rf.fit(X_train_scaled, y_train)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    return importance_df.head(10)['Feature'].tolist()

def process_stock(ticker, horizons=[1, 3]):
    print("Test: Entering process_stock")
    """Process a single stock."""
    try:
        logger.info(f"Starting processing for {ticker}")
        top_features = getStockFeatures(ticker)
        if not top_features:
            logger.error(f"No features returned for {ticker}.")
            return None
        results = []
        model_path = f"C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/combined/{ticker}_combined.parquet"
        target_col = f"{ticker}_Close"
        for horizon in horizons:
            logger.info(f"Training model for {ticker} horizon {horizon}")
            use_lstm = (horizon == 3)  # Use LSTM for 3-month horizon
            model, metrics, backtest_df = train_pytorch_forecast(model_path, target_col, top_features, horizon, use_lstm=use_lstm)
            if metrics is None:
                logger.error(f"Failed to train model for {ticker} horizon {horizon}")
                continue
            results.append({
                'ticker': ticker,
                'horizon': horizon,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'top_features': top_features,
                'backtest_df': backtest_df
            })
        logger.info(f"Completed processing for {ticker} with {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None

def create_dashboard(results):
    """Create a Dash dashboard for buy/sell signals."""
    app = Dash(__name__)
    children = [
        html.H1("Stock Price Prediction Dashboard"),
        html.H2("Top Stocks by R²")
    ]
    
    for result in results:
        ticker = result['ticker']
        horizon = result['horizon']
        backtest_df = result['backtest_df']
        fig = px.line(backtest_df, x="Date", y=["Actual", "Predicted"], title=f"{ticker} {horizon}-Month Forecast")
        fig.add_scatter(x=backtest_df[backtest_df["Signal"] == "Buy"]["Date"],
                        y=backtest_df[backtest_df["Signal"] == "Buy"]["Predicted"],
                        mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=10, color="green"))
        fig.add_scatter(x=backtest_df[backtest_df["Signal"] == "Sell"]["Date"],
                        y=backtest_df[backtest_df["Signal"] == "Sell"]["Predicted"],
                        mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=10, color="red"))
        children.append(html.H3(f"{ticker} (Horizon: {horizon} months, R²: {result['r2']:.3f})"))
        children.append(dcc.Graph(figure=fig))
        children.append(dash_table.DataTable(
            data=backtest_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in backtest_df.columns],
            style_table={'overflowX': 'auto'},
            page_size=10
        ))
        children.append(html.Hr())

    app.layout = html.Div(children)
    app.run(debug=True)

if __name__ == "__main__":
    print("Test: Main block")
    tickers = pd.read_csv("C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/src/data/equities.csv")['Symbol'].tolist()
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    # Process a single ticker for testing
    results = process_stock("AAPL")
    print(f"Results: {results}")
    
    if results:
        # Create dashboard for AAPL
        create_dashboard(results)
        
        # Process all tickers
        results = Parallel(n_jobs=-1)(delayed(process_stock)(ticker) for ticker in tickers)  # Limit for demo
        results = [r for r in sum([r if r else [] for r in results], []) if r is not None]
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            logger.info(f"Results DataFrame shape: {results_df.shape}")
            top_10 = results_df.sort_values(by='r2', ascending=False).head(10)
            logger.info(f"Top 10 models by R²:\n{top_10[['ticker', 'horizon', 'r2', 'rmse', 'mae']]}")
            top_10.to_csv("top_10_forecasts_by_r2.csv", index=False)

            with mlflow.start_run(run_name="Top_10_Forecasts_Summary"):
                mlflow.log_artifact("top_10_forecasts_by_r2.csv")
                for i, row in top_10.iterrows():
                    mlflow.log_metric(f"{row['ticker']}_horizon{row['horizon']}_r2", row['r2'])
                    mlflow.log_metric(f"{row['ticker']}_horizon{row['horizon']}_rmse", row['rmse'])
                    mlflow.log_metric(f"{row['ticker']}_horizon{row['horizon']}_mae", row['mae'])

            create_dashboard(results)  # Dashboard for all tickers