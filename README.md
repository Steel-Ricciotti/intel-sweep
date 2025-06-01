# 📈 Stock Price Prediction with Macroeconomic Indicators 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Databricks](https://img.shields.io/badge/Azure%20Databricks-9.1-red) ![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

Welcome to **Intel-Sweep**, a cutting-edge machine learning pipeline that predicts stock prices using macroeconomic indicators from FRED and financial data from Yahoo Finance. Built on Azure Databricks, this project leverages MLflow, Spark, and Dash to deliver robust forecasting models, interactive dashboards, and professional-grade visualizations. Whether you're a data scientist or a finance enthusiast, this repo showcases an end-to-end ML workflow with real-world impact. 🎥 *Check out the demo video for a live walkthrough!*

## 🎯 Project Overview

Intel-Sweep aims to forecast stock and asset prices (e.g., DE, CAT, EMR, BTC-USD, GLD, VNQ, XLY) by modeling their relationship with macroeconomic indicators like Average Weekly Hours of Manufacturing (AWHMAN), M2 Money Supply (M2SL), Consumer Price Index (CPI), and Unemployment Rate (UNRATE). Using advanced ML models (LSTM, XGBoost, LinearRegression), we generate trading signals, evaluate performance with metrics (R², RMSE, MAE), and visualize results through interactive Dash dashboards and Plotly HTML plots.

## 📊 Data Sources

- **FRED (Federal Reserve Economic Data)**:
  - Macroeconomic indicators fetched via the `fredapi` Python library.
  - Key indicators:
    - **AWHMAN**: Average Weekly Hours in Manufacturing (impacts industrial stocks like DE, CAT, EMR).
    - **M2SL**: M2 Money Supply (influences assets like BTC-USD, GLD, SLV).
    - **CPI**: Consumer Price Index (affects real estate ETFs like VNQ).
    - **UNRATE**: Unemployment Rate (correlates with consumer discretionary ETFs like XLY).
- **Yahoo Finance**:
  - Historical price data for stocks and assets using `yfinance`.
  - Sample Tickers: DE, CAT, EMR, BTC-USD, GLD, SLV, VNQ, XLY, COP, CVX, XLI, XOM.

## 🛠️ Pipeline Breakdown

1. **Data Ingestion**:
   - Fetch FRED indicators and Yahoo Finance price data.
   - Store raw data in Azure Databricks DBFS (`/FileStore/raw/`).
2. **Preprocessing**:
   - Merge indicator and price data on date.
   - Engineer features (e.g., lagged prices, moving averages).
   - Save processed data as Delta tables in `/FileStore/outputs/results/`.
3. **Modeling**:
   - Train models: LSTM, XGBoost, LinearRegression.
   - Use Spark for scalable data processing.
   - Log experiments in MLflow
4. **Backtesting**:
   - Generate buy/sell signals based on predictions.
   - Compute metrics: R², RMSE, MAE.
   - Save backtest results as Delta tables (`backtest_DE_LSTM/`, etc.).
5. **Visualization**:
   - Create interactive Dash dashboards with nested tabs for each ticker and model.
   - Export Plotly HTML plots (`DE_LSTM_backtest_plot.html`) to `/FileStore/outputs/`.
   - Summarize metrics in `/FileStore/outputs/metrics_summary.csv`.

## 🧪 Experiments

The project runs eight experiments, each focusing on a macroeconomic indicator and related assets:

1. **AWHMAN**: Predicts industrial stocks (DE, CAT, EMR) using manufacturing hours.
2. **M2SL**: Forecasts crypto and commodities (BTC-USD, GLD, SLV) with money supply.
3. **CPI**: Models real estate (VNQ) based on inflation.
4. **UNRATE**: Predicts consumer discretionary (XLY) using unemployment data.
5. **Bitcoin/Gold** Predicts the price of gold based on M2/Money Supply
6. **S&P500** Predicts future S&P500 prices based on an ensemble of indicators.
7. **Oil** Predicts the performance of key oil companies based on oil prices.
8. **Banks** Predicts the performance of several bank stocks based on interest rates and yield curves.


Each experiment is tracked in MLflow with dedicated paths and visualized via custom Dash apps.

## 💻 Tech Stack

- **Azure Databricks**: Scalable compute and storage for data processing and ML.
- **MLflow**: Tracks experiments, models, and metrics.
- **Dash & Plotly**: Powers interactive dashboards and HTML plots.
- **Python Libraries**: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `xgboost`, `fredapi`, `yfinance`.
- **DBFS**: Stores raw data, Delta tables, and outputs in `/FileStore/`.

## 📈 Key Outputs

- **Delta Tables**: Backtest results in `/FileStore/outputs/results/backtest_<TICKER>_<MODEL>/` (e.g., `backtest_DE_LSTM/`).
- **HTML Plots**: Visualizations in `/FileStore/outputs/<TICKER>_<MODEL>_backtest_plot.html`.
- **Metrics**: R², RMSE, MAE logged in MLflow and summarized in `/FileStore/outputs/metrics_summary.csv`.
- **Dashboards**: Interactive Dash apps for each experiment, showing predictions and signals.

## 🚀 Getting Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/intel-sweep.git
   ```
2. **Set Up Databricks**:
   - Import notebooks (`workweek_industrials.py`, `m2_equities_workflow.py`, etc.).
   - Configure DBFS paths and MLflow tracking.
3. **Install Dependencies**:
   ```python
   %pip install pandas numpy scikit-learn tensorflow xgboost fredapi yfinance dash plotly mlflow
   ```
4. **Run Experiments**:
   - Execute scripts like `workweek_industrials.py` to generate Delta tables and plots.
   - Use `render_awhman_dashboard.py` to launch Dash apps.
5. **Explore Outputs**:
   - View HTML plots in `/FileStore/outputs/`.
   - Check metrics in MLflow or `metrics_summary.csv`.
   - Interact with Dash dashboards for each experiment.

## 🎥 Demo Video

Watch the [demo video](https://your-video-link.com) to see the pipeline in action! From fetching FRED and Yahoo Finance data to training models and rendering Dash dashboards, it’s an end-to-end showcase of predictive ML in action.

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙌 Acknowledgments

- **FRED** and **Yahoo Finance** for open data access.
- **Azure Databricks** for a powerful ML platform.
- **MLflow** and **Dash** for seamless tracking and visualization.

---
*Ready to predict the market? Dive into Intel-Sweep and start forecasting! 🌟*
