# Cell 1: Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import plotly.express as px

# Cell 2: Load Data
data_path = "/dbfs/FileStore/combined/AAPL_combined.parquet"
ticker = "AAPL"  # Define the ticker symbol
data_path = f"C:/Users/Steel/Desktop/Projects/intel-sweep/intel-sweep/data/combined/{ticker}_combined.parquet"
df = pd.read_parquet(data_path)
target = "AAPL_Close"
lag_features = [col for col in df.columns if "_lag" in col]  # Only _lag features
print(f"Lag Features ({len(lag_features)}):", lag_features)

# Prepare data
X = df[lag_features].fillna(0)  # Handle NaNs
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cell 3: Combined R-squared (All Lag Features)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
combined_r2 = rf.score(X_test_scaled, y_test)
print(f"Combined R-squared (all {len(lag_features)} lag features): {combined_r2:.3f}")

# Cell 4: Test Feature Combinations
max_features = min(5, len(lag_features))  # Limit for speed
best_r2 = 0
best_combo = None
results = []

# Try combinations of 1 to max_features
for r in range(1, max_features + 1):
    for combo in combinations(range(len(lag_features)), r):
        subset = [lag_features[i] for i in combo]
        X_train_subset = X_train_scaled[:, list(combo)]
        X_test_subset = X_test_scaled[:, list(combo)]
        rf.fit(X_train_subset, y_train)
        r2 = rf.score(X_test_subset, y_test)
        results.append({"Features": ", ".join(subset), "R-squared": r2})
        if r2 > best_r2:
            best_r2 = r2
            best_combo = subset

# Cell 5: Display Results
results_df = pd.DataFrame(results).sort_values(by="R-squared", ascending=False).head(10)
print("Top 10 Feature Combinations by R-squared:")
display(results_df)

print(f"Best Combination: {best_combo}")
print(f"Best R-squared: {best_r2:.3f}")

# Cell 6: Visualize Top Combinations
fig = px.bar(results_df.head(10), x="R-squared", y="Features", orientation="h",
             title="Top 10 Lag Feature Combinations by R-squared")
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.show()