import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.pytorch
import logging
from pathlib import Path
import numpy as np
from src.data.prep_data import prepare_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_custom_metrics(y_true, y_pred):
    """Calculate custom metrics for model performance."""
    bias = np.mean(y_pred - y_true)
    directional_accuracy = np.mean(np.sign(y_pred[1:] - y_true[:-1]) == np.sign(y_true[1:] - y_true[:-1]))
    return {"bias": bias, "directional_accuracy": directional_accuracy}

def monitor_model(model_name, model_version, data_file, fred_files):
    """Monitor model performance on new data."""
    # Load model
    model_uri = f"models:/{model_name}/{model_version}"
    if "XGBoost" in model_name:
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.pytorch.load_model(model_uri)

    # Prepare data
    X, _, y, _, _ = prepare_data(stock_ticker='AAPL', fred_files=fred_files, stock_file=data_file, test_size=0.0)

    # Predict
    if "XGBoost" in model_name:
        y_pred = model.predict(X)
    else:
        from sklearn.preprocessing import StandardScaler
        import torch
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor).numpy().flatten()

    # Calculate metrics
    metrics = calculate_custom_metrics(y, y_pred)
    logger.info(f"Monitoring metrics: {metrics}")

    # Log to MLflow
    with mlflow.start_run(run_name="model_monitoring"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("data_file", str(data_file))

    return metrics

if __name__ == "__main__":
    fred_files = [
        Path("data/fred_economic_indicators/00XALCATM086NEST.csv"),
        Path("data/fred_economic_indicators/00XALCBEM086NEST.csv"),
        Path("data/fred_economic_indicators/00XALCCZM086NEST.csv")
    ]
    stock_file = Path("data/av_monthly/AAPL_monthly.csv")
    mlflow.set_experiment("stock_price_prediction")
    metrics = monitor_model("StockPriceXGBoost", "1", stock_file, fred_files)
    print(metrics)