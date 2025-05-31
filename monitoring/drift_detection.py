import pandas as pd
import logging
import mlflow
from pathlib import Path
from scipy.stats import ks_2samp
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os 
# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_drift(data_file, column, reference_end_date='2024-12-31', window_months=12, threshold=0.05):
    """
    Detect data drift using KS test with a rolling window for monthly updates.
    
    Args:
        data_file (Path or str): Path to FRED data CSV.
        column (str): Column to test for drift (e.g., '00XALCATM086NEST').
        reference_end_date (str): End date for reference data (YYYY-MM-DD).
        window_months (int): Size of current data window in months.
        threshold (float): P-value threshold for drift detection.
    
    Returns:
        dict: KS statistic, p-value, and drift status.
    """
    # Load data
    if IS_DATABRICKS:
        spark = SparkSession.builder.appName("DriftDetection").getOrCreate()
        df = spark.read.csv(str(data_file), header=True, inferSchema=True).toPandas()
    else:
        df = pd.read_csv(data_file)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Define reference and current windows
    ref_end = pd.to_datetime(reference_end_date)
    ref_data = df[df['date'] <= ref_end][column].dropna()
    
    latest_date = df['date'].max()
    window_start = latest_date - relativedelta(months=window_months - 1)
    curr_data = df[(df['date'] >= window_start) & (df['date'] <= latest_date)][column].dropna()

    if len(curr_data) < 6:
        logger.warning(f"Current window has {len(curr_data)} samples, too few for KS test")
        return {"ks_statistic": None, "p_value": None, "drift_detected": False}

    # Perform KS test
    ks_stat, p_value = ks_2samp(ref_data, curr_data)
    drift_detected = p_value < threshold

    logger.info(f"KS Test for {column}: statistic={ks_stat:.4f}, p-value={p_value:.4f}, drift={'detected' if drift_detected else 'not detected'}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"drift_{column}_{latest_date.strftime('%Y%m%d')}"):
        mlflow.log_metric("ks_statistic", ks_stat)
        mlflow.log_metric("p_value", p_value)
        mlflow.log_param("column", column)
        mlflow.log_param("drift_detected", drift_detected)
        mlflow.log_param("reference_end_date", reference_end_date)
        mlflow.log_param("window_months", window_months)
        mlflow.log_param("latest_date", latest_date.strftime('%Y-%m-%d'))

    return {"ks_statistic": ks_stat, "p_value": p_value, "drift_detected": drift_detected}

if __name__ == "__main__":
    mlflow.set_experiment("stock_price_prediction")
    data_file = Path("src/data/fred_economic_indicators/00XALCATM086NEST.csv") if not IS_DATABRICKS else "/mnt/fred_data/00XALCATM086NEST.csv"
    result = detect_drift(data_file, column="00XALCATM086NEST")
    print(result)