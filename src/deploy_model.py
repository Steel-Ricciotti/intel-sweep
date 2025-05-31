import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# Get best model version (e.g., lowest RMSE)
experiment = mlflow.get_experiment_by_name("stock_price_prediction")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
model_name = "StockPriceXGBoost" if "xgboost" in best_run.data.tags.get("mlflow.log-model.history", "") else "StockPricePyTorch"
model_version = client.get_latest_versions(model_name, stages=["None"])[0].version

# Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production",
    archive_existing_versions=True
)

# Create serving endpoint
endpoint_name = "stock-price-predictor"
client.create_model_serving_endpoint(
    name=endpoint_name,
    model_name=model_name,
    model_version=model_version,
    workload_size="Small",
    scale_to_zero_enabled=True
)

print(f"Model {model_name} version {model_version} deployed to endpoint {endpoint_name}")