# Databricks notebook source
# COMMAND ----------
# Import libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
%matplotlib inline

# COMMAND ----------
# Load combined data
ticker = "SPY"
data_path = f"/dbfs/FileStore/combined/{ticker}_combined.parquet"
df = pd.read_parquet(data_path)
display(df.head())

# COMMAND ----------
# Plot time series for key indicators
fig = make_subplots(rows=2, cols=1, subplot_titles=["SPY Close vs. VIX (Lagged)", "CPI vs. Stock Price"])
fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_Close"], name="SPY Close"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["VIXCLS_lag3"], name="VIXCLS (3-month lag)"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_Close"], name="SPY Close"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["CPIAUCSL_lag3"], name="CPI (3-month lag)"), row=2, col=1)
fig.update_layout(title="Time Series Analysis", height=600)
display(fig)

# COMMAND ----------
# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Features")
display(plt)

# COMMAND ----------
# Stationarity test (ADF test) for SPY_Close
result = adfuller(df[f"{ticker}_Close"].dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] < 0.05:
    print("Series is stationary (reject null hypothesis)")
else:
    print("Series is non-stationary (consider differencing)")

# COMMAND ----------
# Feature importance preview (mock for EDA)
from sklearn.ensemble import RandomForestRegressor
X = df.drop(columns=[f"{ticker}_Close", f"{ticker}_Close_ma3", f"{ticker}_Close_ma12"])
y = df[f"{ticker}_Close"]
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig = px.bar(importance.head(10), title="Top 10 Feature Importance (Random Forest)")
display(fig)