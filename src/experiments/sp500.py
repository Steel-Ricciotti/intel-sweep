# Leading Economic Indicators on FRED
# Yield Curve (10-Year Treasury Constant Maturity Minus 2-Year)

# FRED Series ID: T10Y2Y

# Inverted yield curves often precede recessions and stock downturns.

# Initial Unemployment Claims (Weekly)

# FRED Series ID: ICSA

# A rise often signals weakening labor markets and future stock weakness.

# ISM Manufacturing New Orders Index

# FRED Proxy: NAPMNOI

# New orders are a forward-looking component of industrial activity.

# Building Permits: New Private Housing Units

# FRED Series ID: PERMIT

# A signal of future construction activity and broader economic health.

# Consumer Sentiment Index (University of Michigan)

# FRED Series ID: UMCSENT

# High sentiment often precedes strong consumer spending and corporate earnings.

# S&P 500 Index (itself) – as a component of the Leading Economic Index

# FRED Series ID: SP500

# The market often anticipates future economic activity and earnings.

# Manufacturers’ New Orders for Consumer Goods

# FRED Series ID: ACOGNO

# Reflects anticipated consumer demand.

# Money Supply (M2)

# FRED Series ID: M2SL

# Rapid changes can signal shifts in liquidity and monetary policy impacts.

# Average Weekly Hours of Production Workers (Manufacturing)

# FRED Series ID: AWHAEMAN

# Changes often precede broader employment and output shifts.

# Leading Index for the United States

# FRED Series ID: USSLIND

# A composite index designed to signal turning points in the business cycle.


leading_indicators = ['USSLIND','WTISPLC','PERMIT','UMCSENT','M2SL']
equities = ['SPY']


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