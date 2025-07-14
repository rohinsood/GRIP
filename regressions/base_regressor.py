from abc import ABC, abstractmethod
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BaseRegressor(ABC):
    def __init__(self, data, target_column, config, experiment_folder_name, use_invariance=True):
        self.data = data
        self.target_column = target_column
        self.config = config
        self.experiment_folder_name = experiment_folder_name
        self.results_path = f"results/model/{experiment_folder_name}/"
        self.use_invariance = use_invariance  # ✅ add invariance flag

    def prepare_environment_data(self):
        data = self.data.copy()
        target_column = self.target_column
        drop_columns = [target_column, "Sample name", "Environment"]

        raw_data = data[~data["Environment"].str.contains("_")]
        non_raw_data = data[data["Environment"].str.contains("_")]

        from sklearn.model_selection import train_test_split
        raw_train, raw_test = train_test_split(raw_data, test_size=0.2, random_state=42)
        train_data = pd.concat([non_raw_data, raw_train], ignore_index=True)
        test_data = raw_test

        return data, drop_columns, target_column, train_data, test_data

    @abstractmethod
    def train_and_evaluate(self):
        pass

    def save_results(self, mse, rmse, mae, r2, r, top_features, model_type):
        top_features_str = "; ".join([f"{feat}:{imp:.6f}" for feat, imp in top_features])
        df = pd.DataFrame({
            "mse": [mse],
            "rmse": [rmse],
            "mae": [mae],
            "r2": [r2],
            "r": [r],
            "top_features": [top_features_str]
        })
        file_path = self.results_path + model_type + ".csv"
        df.to_csv(file_path, index=False)
        return df, file_path

    def plot_residuals(self, y_true, y_pred, X, top_features, dataset_label, model_type):
        residuals = y_true - y_pred
        graph_dir = f"results/graphs/{self.experiment_folder_name}/{model_type}_{dataset_label}"
        os.makedirs(graph_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(f"Predicted ({dataset_label})")
        plt.ylabel("Residuals")
        plt.title(f"{dataset_label} Residuals vs. Predicted")
        plt.tight_layout()
        plt.savefig(f"{graph_dir}/residuals_vs_predicted.png")
        plt.close()

        if top_features:
            top_feature_name = top_features[0][0]
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=X[top_feature_name], y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel(f"{top_feature_name} ({dataset_label})")
            plt.ylabel("Residuals")
            plt.title(f"{dataset_label} Residuals vs. {top_feature_name}")
            plt.tight_layout()
            plt.savefig(f"{graph_dir}/residuals_vs_{top_feature_name}.png")
            plt.close()
