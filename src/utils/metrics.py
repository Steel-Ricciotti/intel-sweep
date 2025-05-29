import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE and MAE.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
    
    Returns:
        dict: Metrics dictionary.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}