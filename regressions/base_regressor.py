from abc import ABC, abstractmethod
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

class BaseRegressor(ABC):
    def __init__(self, data, target_column, config, experiment_folder_name):
        self.data = data
        self.target_column = target_column
        self.config = config
        self.experiment_folder_name = experiment_folder_name
        self.results_path = f"results/{experiment_folder_name}/"

    def prepare_environment_data(self):
        data = self.data.copy()
        target_column = self.target_column
        drop_columns = [target_column, "Sample name", "Environment"]

        raw_data = data[~data["Environment"].str.contains("_")]
        non_raw_data = data[data["Environment"].str.contains("_")]

        raw_train, raw_test = train_test_split(raw_data, test_size=0.2, random_state=42)

        train_data = pd.concat([non_raw_data, raw_train], ignore_index=True)
        test_data = raw_test

        return data, drop_columns, target_column, train_data, test_data


    @abstractmethod
    def train_and_evaluate(self):
        pass
    
    
    def save_results(self, mse, rmse, mae, r2, r, top_features, model_type):
        # Convert top_features list of tuples to a string for saving
        top_features_str = "; ".join([f"{feat}:{imp:.6f}" for feat, imp in top_features])

        # Build a DataFrame with a single row
        df = pd.DataFrame({
            "mse": [mse],
            "rmse": [rmse],
            "mae": [mae],
            "r2": [r2],
            "r": [r],
            "top_features": [top_features_str]
        })

        # Example: save to CSV (adjust path as needed)
        # You can customize file path logic elsewhere
        df.to_csv(self.results_path+model_type+".csv", index=False)

        return df
