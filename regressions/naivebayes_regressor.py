from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from .base_regressor import BaseRegressor

class NaiveBayesRegressor(BaseRegressor):

    def train_and_evaluate(self, n_bins=10):
        data = self.data.copy()
        target_column = self.target_column
        drop_columns = [target_column, "Sample name", "Environment"]

        raw_data = data[~data["Environment"].str.contains("_")]
        non_raw_data = data[data["Environment"].str.contains("_")]

        raw_train, raw_test = train_test_split(raw_data, test_size=0.2, random_state=42)
        train_data = pd.concat([non_raw_data, raw_train], ignore_index=True)
        test_data = raw_test

        val_data, train_data = train_test_split(
            train_data, train_size=0.1625, random_state=42, stratify=train_data["Environment"]
        )

        scaler = StandardScaler()
        gnb = GaussianNB()

        # Discretize target
        def discretize_target(y):
            bins = np.linspace(y.min(), y.max(), n_bins + 1)
            labels = np.digitize(y, bins) - 1
            return labels, bins

        environments = train_data["Environment"].unique()
        for e_train in environments:
            train_env = train_data[train_data["Environment"] == e_train]
            X_train = train_env.drop(columns=drop_columns)
            y_train = train_env[target_column]
            y_train_disc, bins = discretize_target(y_train)

            X_train_scaled = scaler.fit_transform(X_train)
            gnb.fit(X_train_scaled, y_train_disc)

            for e_val in environments:
                if e_val == e_train:
                    continue
                val_env = train_data[train_data["Environment"] == e_val]
                X_val = val_env.drop(columns=drop_columns)
                y_val = val_env[target_column]
                X_val_scaled = scaler.transform(X_val)

                y_val_pred_class = gnb.predict(X_val_scaled)
                y_val_pred_reg = 0.5 * (bins[y_val_pred_class] + bins[y_val_pred_class + 1])
                mse = mean_squared_error(y_val, y_val_pred_reg)
                print(f"Validation MSE after training on {e_train}, tested on {e_val}: {mse:.4f}")

        # Final test evaluation
        X_test = test_data.drop(columns=drop_columns)
        y_test = test_data[target_column]
        y_test_disc, bins = discretize_target(y_test)

        X_test_scaled = scaler.transform(X_test)
        y_test_pred_class = gnb.predict(X_test_scaled)
        y_test_pred_reg = 0.5 * (bins[y_test_pred_class] + bins[y_test_pred_class + 1])

        mse = mean_squared_error(y_test, y_test_pred_reg)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred_reg)
        r2 = r2_score(y_test, y_test_pred_reg)
        r = np.sqrt(r2) if r2 >= 0 else float("nan")

        result = permutation_importance(
            gnb, X_test_scaled, y_test_disc, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean
        feature_importance = dict(zip(X_test.columns, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        results_df = self.save_results(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            r=r,
            top_features=top_features,
            model_type="naive_bayes"
        )

        return results_df
