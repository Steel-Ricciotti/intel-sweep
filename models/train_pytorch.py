import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.data.prep_data import prepare_data
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPriceNN(nn.Module):
    """Simple feed-forward neural network for stock price prediction."""
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

def train_pytorch():
    """Train and evaluate a PyTorch neural network with MLflow logging."""
    # Data preparation
    fred_files = [
        Path("src/data/fred_economic_indicators/00XALCATM086NEST.csv"),
        Path("src/data/fred_economic_indicators/00XALCBEM086NEST.csv"),
        Path("src/data/fred_economic_indicators/00XALCCZM086NEST.csv")
    ]
    stock_file = Path("src/data/av_monthly/AAPL_monthly.csv")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(fred_files=fred_files, stock_file=stock_file)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Model parameters
    input_size = X_train.shape[1]
    params = {
        'hidden_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 16
    }

    # Start MLflow run
    with mlflow.start_run() as run:
        # Initialize model, loss, optimizer
        model = StockPriceNN(input_size, params['hidden_size'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Training loop
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
                logger.info(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {loss.item():.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy().flatten()
        metrics = calculate_metrics(y_test, y_pred)
        logger.info(f"Metrics: {metrics}")

        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "pytorch_model")
        mlflow.log_param("features", feature_names)
        # Register model
        model_uri = f"runs:/{run.info.run_id}/pytorch_model"
        registered_model = mlflow.register_model(model_uri, "StockPricePyTorch")
        logger.info(f"Registered model: {registered_model.name} version {registered_model.version}")

    return model, metrics

if __name__ == "__main__":
    mlflow.set_experiment("stock_price_prediction")
    model, metrics = train_pytorch()
    print(f"Model metrics: {metrics}")