leading_indicators = ['UMDMNO']
equities = ['CAT','EMR','FDX','UPS']


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.classes.ETL2 import ETL
from src.classes.features import FeatureEngineer
from src.models.models import Models
from src.monitoring.monitoring import Monitoring
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    # leading_indicators = ['BAA','BOPGSTB','BUSINV','CES0500000003','CFNAI','CIVPART','CPIAUCSL','CSUSHPINSA','DCOILWTICO','DGORDER','DGS10','FEDFUNDS','GDP','GDPC1','HOUST','ICSA','INDPRO','M2SL','MORTGAGE30US','MTSDS133FMS','PAYEMS','PCEPI','PERMIT','PI','PPIACO','RSXFS','SP500','SRVPRD','T10Y2Y','TB3MS','TCU','TEDRATE','UMCSENT','UNRATE','USSLIND','VIXCLS']
    lag_periods = [1, 3, 6]
    ma_windows = [3, 12]
    horizon = 1
    results = []
    data_dir = "C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data"
    output_dir = "C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/outputs"

    etl = ETL(data_dir=data_dir)
    feature_engineer = FeatureEngineer(lag_periods=lag_periods, ma_windows=ma_windows)
    models = Models()
    monitoring = Monitoring(output_dir=output_dir)

    for equity in equities:
        logger.info(f"Processing {equity}")
        # Run ETL pipeline
        data = etl.run(stock_list=[equity], indicators=leading_indicators, experiment_name=f"m2_{equity.lower()}")
        logger.info(f"Merged data shape for {equity}: {data.shape}")
        print(data.head())
        # Engineer features
        combined_data = feature_engineer.transform(data, f'{equity}_Close')
        logger.info(f"Engineered data shape for {equity}: {combined_data.shape}")
        
        # Define features
        # indicator = "M2SL"
        features = [ f"{equity}_Close_ma3", f"{equity}_Close_ma12", f"{equity}_Volume_lag1", f"{equity}_Close_volatility"]

        for i in leading_indicators:
            features.append(i)
            features.append(f"{i}_lag1")
            features.append(f"{i}_lag3")
            features.append(f"{i}_lag6")
        # Train models
        for model_type in ['LSTM', 'XGBoost', 'LinearRegression']:
            logger.info(f"Training {model_type} for {equity}")
            if model_type == 'LSTM':
                model, metrics, backtest_df = models.train_pytorch_forecast(combined_data, f"{equity}_Close", features, horizon)
            elif model_type == 'XGBoost':
                model, metrics, backtest_df = models.train_xgboost(combined_data, f"{equity}_Close", features, horizon)
            elif model_type == 'LinearRegression':
                model, metrics, backtest_df = models.train_linear_regression(combined_data, f"{equity}_Close", features, horizon)
            
            if backtest_df is not None:
                logger.info(f"{model_type} metrics for {equity}: {metrics}")
                results.append({
                    'ticker': f"{equity}_{model_type}",
                    'horizon': horizon,
                    'metrics': metrics,
                    'backtest_df': backtest_df
                })

    # if results:
    #     monitoring.create_dashboard(results)