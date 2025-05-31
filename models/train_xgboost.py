import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import logging
from pathlib import Path
from src.data.prep_data import prepare_data
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_xgboost():
    """Train and evaluate an XGBoost model with MLflow logging."""
    # Data preparation
    fred_files = [
        Path("src/data/fred_economic_indicators/00XALCATM086NEST.csv"),
        Path("src/data/fred_economic_indicators/00XALCBEM086NEST.csv"),
        Path("src/data/fred_economic_indicators/00XALCCZM086NEST.csv")
    ]
    stock_file = Path("src/data/av_monthly/AAPL_monthly.csv")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(fred_files=fred_files, stock_file=stock_file)

    # Model parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    }

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Trained XGBoost model")

        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        logger.info(f"Metrics: {metrics}")

        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_param("features", feature_names)

        # Register model
        model_uri = f"runs:/{run.info.run_id}/xgboost_model"
        registered_model = mlflow.register_model(model_uri, "StockPriceXGBoost")
        logger.info(f"Registered model: {registered_model.name} version {registered_model.version}")

    return model, metrics

if __name__ == "__main__":
    mlflow.set_experiment("stock_price_prediction")
    model, metrics = train_xgboost()
    print(f"Model metrics: {metrics}")