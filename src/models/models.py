import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.pytorch
import logging
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPriceLSTM(nn.Module):
    """LSTM network for stock price forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        """
        Initialize LSTM model.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of LSTM hidden units.
            num_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        mlflow.set_tracking_uri("http://localhost:5000")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class Models:
    """Manages ML models for stock price forecasting."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate RMSE, MAE, and R²."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def train_linear_regression_forecast():
        """
        Placeholder for XGBoost training method.
        Implement this method to train a linear regression model for forecasting.
        """
        pass

    def train_xgboost(
            self,
            data: pd.DataFrame,
            target_col: str,
            features: List[str],
            horizon: int = 1,
            params: Optional[Dict[str, Any]] = None
        ) -> Tuple[Optional[xgb.XGBRegressor], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
            """
            Train an XGBoost model for stock price forecasting.
            
            Args:
                data (pd.DataFrame): Input data with features and target.
                target_col (str): Target column (e.g., 'BTC-USD_Close').
                features (List[str]): List of feature columns.
                horizon (int): Forecasting horizon in months.
                params (Dict[str, Any], optional): Model hyperparameters.
            
            Returns:
                Tuple: (model, metrics_dict, backtest_df) or (None, None, None) on failure.
            """
            logger.info(f"Starting XGBoost training for {target_col}, horizon {horizon}")
            if params is None:
                params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'objective': 'reg:squarederror'
                }

            try:
                df = data.copy()
                logger.info(f"Training data shape: {df.shape}")
                if target_col not in df.columns:
                    logger.error(f"Target column {target_col}: not found in data")
                    return None, None, None

                # Validate features
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    logger.error(f"Missing features: {missing_features}")
                    return None, None, None

                # Create target for forecasting
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                df[f'{target_col}_t{horizon}'] = df[target_col].shift(-horizon)
                df = df.dropna(subset=features + [f'{target_col}_t{horizon}'])
                logger.info(f"Data shape after NaN handling: {df.shape}")
                if df.empty:
                    logger.error(f"No data for stock {target_col} after dropping rows with NaN values")
                    return None, None, None

                # Split data
                X = df[features]
                y = df[f'{target_col}_t{horizon}']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model = xgb.XGBRegressor(**params)
                model.fit(X_train_scaled, y_train)

                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                metrics = self.calculate_metrics(y_test, y_pred)
                logger.info(f"Metrics for {target_col}: {metrics}")

                # Generate backtest DataFrame
                current_prices = df.loc[y_test.index, target_col]
                if len(y_pred) != len(current_prices):
                    logger.error(f"Shape mismatch: y_pred ({len(y_pred)}), current_prices ({len(current_prices)})")
                    return None, None, None

                price_change = (y_pred - current_prices) / current_prices * 100
                signals = np.where(price_change > 5, "Buy", np.where(price_change < -5, "Sell", ""))
                backtest_df = pd.DataFrame({
                    "Date": y_test.index,
                    "Actual": y_test,
                    "Predicted": y_pred,
                    "Current_Price": current_prices,
                    "Signal": signals
                })
                backtest_path = f"backtest_{target_col}_XGBoost_horizon_{horizon}.csv"
                backtest_df.to_csv(backtest_path, index=False)

                # Log to MLflow
                mlflow.set_experiment("stock_price_forecasting")
                with mlflow.start_run(run_name=f"{target_col}_XGBoost_horizon_{horizon}"):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("stock_id", target_col.split('_')[0])
                    mlflow.set_tag("horizon", horizon)
                    mlflow.set_tag("model_type", "XGBoost")
                    mlflow.log_param("features", ", ".join(features))
                    mlflow.log_artifact(backtest_path)
                    mlflow.xgboost.log_model(model, "xgboost_model", input_example=X_train_scaled[:1])
                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgboost_model"
                    registered_model = mlflow.register_model(model_uri, f"StockPriceXGBoost_Horizon_{horizon}")
                    logger.info(f"Model registered: {registered_model.name} version {registered_model.version}")

                return model, metrics, backtest_df
            except Exception as e:
                logger.error(f"Error in XGBoost training for {target_col}: {str(e)}")
                return None, None, None
            
        
    def train_pytorch_forecast(
        self,
        data: pd.DataFrame,
        target_col: str,
        features: List[str],
        horizon: int = 1,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[StockPriceLSTM], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
        """
        Train PyTorch LSTM model for h-month-ahead forecasting.
        
        Args:
            data (pd.DataFrame): Input data with features and target.
            target_col (str): Target column (e.g., 'BTC-USD_Close').
            features (List[str]): List of feature columns.
            horizon (int): Forecasting horizon in months.
            params (Dict[str, Any], optional): Model hyperparameters.
        
        Returns:
            Tuple: (model, metrics_dict, backtest_df) or (None, None, None) on failure.
        """
        print(mlflow.get_tracking_uri())
        logger.info(f"Starting training for {target_col}, horizon {horizon}")
        if params is None:
            params = {'hidden_size': 128, 'epochs': 200, 'learning_rate': 0.0005, 'batch_size': 128}

        try:
            df = data.copy()
            logger.info(f"Training data shape: {df.shape}")
            if target_col not in df.columns:
                logger.error(f"Target column {target_col} not found in data")
                return None, None, None

            # Validate features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None, None, None

            # Create target for forecasting
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df[f'{target_col}_t{horizon}'] = df[target_col].shift(-horizon)
            df = df.dropna(subset=features + [f'{target_col}_t{horizon}'])
            logger.info(f"Data shape after NaN handling: {df.shape}")
            if df.empty:
                logger.error(f"No valid data for {target_col} after NaN handling")
                return None, None, None

            # Split data
            X = df[features]
            y = df[f'{target_col}_t{horizon}']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Scale features and target
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = y_test.values

            # Reshape for LSTM [samples, timesteps, features]
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

            # Initialize model
            input_size = X.shape[1]
            model = StockPriceLSTM(input_size, params['hidden_size'])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

            # MLflow experiment
            mlflow.set_experiment("stock_price_forecasting")
            with mlflow.start_run(run_name=f"{target_col}_horizon{horizon}"):
                mlflow.set_tag("ticker", target_col.split('_')[0])
                mlflow.set_tag("horizon", horizon)
                mlflow.set_tag("model_type", "LSTM")

                # Train model
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

                # Evaluate model
                model.eval()
                with torch.no_grad():
                    y_pred_scaled = model(X_test_tensor).numpy().flatten()
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                metrics = self.calculate_metrics(y_test, y_pred)
                logger.info(f"Metrics for {target_col} (horizon {horizon}): {metrics}")

                # Generate backtest DataFrame
                current_prices = df.loc[y_test.index, target_col]
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
                backtest_df.to_csv(f"backtest_{target_col}_horizon{horizon}.csv", index=False)

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("features", ", ".join(features))
                mlflow.log_artifact(f"backtest_{target_col}_horizon{horizon}.csv")
                mlflow.pytorch.log_model(model, "pytorch_model", input_example=X_train_scaled[:1])
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/pytorch_model"
                registered_model = mlflow.register_model(model_uri, f"StockPricePyTorch_Horizon{horizon}")
                logger.info(f"Model registered: {registered_model.name} version {registered_model.version}")

            return model, metrics, backtest_df
        except Exception as e:
            logger.error(f"Error in train_pytorch_forecast for {target_col} horizon {horizon}: {str(e)}")
            return None, None, None
        
    def train_linear_regression(
        self,
        data: pd.DataFrame,
        target_col: str,
        features: List[str],
        horizon: int = 1,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[LinearRegression], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
        """
        Train a Linear Regression model for stock price forecasting.
        
        Args:
            data (pd.DataFrame): Input data with features and target.
            target_col (str): Target column (e.g., 'BTC-USD_Close').
            features (List[str]): List of feature columns.
            horizon (int): Forecasting horizon in months.
            params (Dict[str, Any], optional): Model hyperparameters.
        
        Returns:
            Tuple: (model, metrics_dict, backtest_df) or (None, None, None) on failure.
        """
        logger.info(f"Starting Linear Regression training for {target_col}, horizon {horizon}")
        if params is None:
            params = {'fit_intercept': True}

        try:
            df = data.copy()
            logger.info(f"Training data shape: {df.shape}")
            if target_col not in df.columns:
                logger.error(f"Target column {target_col} not found in data")
                return None, None, None

            # Validate features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None, None, None

            # Create target for forecasting
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df[f'{target_col}_t{horizon}'] = df[target_col].shift(-horizon)
            df = df.dropna(subset=features + [f'{target_col}_t{horizon}'])
            logger.info(f"Data shape after NaN handling: {df.shape}")
            if df.empty:
                logger.error(f"No valid data for {target_col} after NaN handling")
                return None, None, None

            # Split data
            X = df[features]
            y = df[f'{target_col}_t{horizon}']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LinearRegression(**params)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            metrics = self.calculate_metrics(y_test, y_pred)
            logger.info(f"Metrics for {target_col}: {metrics}")

            # Generate backtest DataFrame
            current_prices = df.loc[y_test.index, target_col]
            if len(y_pred) != len(current_prices):
                logger.error(f"Shape mismatch: y_pred ({len(y_pred)}), current_prices ({len(current_prices)})")
                return None, None, None

            price_change = (y_pred - current_prices) / current_prices * 100
            signals = np.where(price_change > 5, "Buy", np.where(price_change < -5, "Sell", ""))
            backtest_df = pd.DataFrame({
                "Date": y_test.index,
                "Actual": y_test,
                "Predicted": y_pred,
                "Current_Price": current_prices,
                "Signal": signals
            })
            backtest_path = f"backtest_{target_col}_LinearRegression_horizon_{horizon}.csv"
            backtest_df.to_csv(backtest_path, index=False)

            # Log to MLflow
            mlflow.set_experiment("stock_price_forecasting")
            with mlflow.start_run(run_name=f"{target_col}_LinearRegression_horizon_{horizon}"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tag("stock_id", target_col.split('_')[0])
                mlflow.set_tag("horizon", horizon)
                mlflow.set_tag("model_type", "LinearRegression")
                mlflow.log_param("features", ", ".join(features))
                mlflow.log_artifact(backtest_path)
                mlflow.sklearn.log_model(model, "linear_model", input_example=X_train_scaled[:1])
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/linear_model"
                registered_model = mlflow.register_model(model_uri, f"StockPriceLinearRegression_Horizon_{horizon}")
                logger.info(f"Model registered: {registered_model.name} version {registered_model.version}")

            return model, metrics, backtest_df
        except Exception as e:
            logger.error(f"Error in Linear Regression training for {target_col}: {str(e)}")
            return None, None, None