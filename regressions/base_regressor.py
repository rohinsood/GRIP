from abc import ABC, abstractmethod
import os
import pandas as pd
from datetime import datetime

class BaseRegressor(ABC):
    def __init__(self, data, target_column, n_features, config):
        self.data = data
        self.target_column = target_column
        self.n_features = n_features
        self.config = config


    @abstractmethod
    def train_and_evaluate(self):
        pass
    
    
    def save_results(self, mse, rmse, mae, r2, r, selected_features, top_features):
        # Convert top_features list of tuples to a string for saving
        top_features_str = "; ".join([f"{feat}:{imp:.6f}" for feat, imp in top_features])

        # Build a DataFrame with a single row
        df = pd.DataFrame({
            "mse": [mse],
            "rmse": [rmse],
            "mae": [mae],
            "r2": [r2],
            "r": [r],
            "selected_features": [",".join(selected_features)],
            "top_features": [top_features_str]
        })

        # Example: save to CSV (adjust path as needed)
        # You can customize file path logic elsewhere
        df.to_csv(self.results_path, index=False)

        return df
