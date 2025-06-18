from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from .base_regressor import BaseRegressor

class RandomforestRegressor(BaseRegressor):
    
    def train_and_evaluate(self):
        data = self.data.copy()
        target_column = self.target_column
        drop_columns = [target_column, "Sample name", "Environment"]

        # Basic split into train/test sets
        raw_data = data[~data["Environment"].str.contains("_")]
        non_raw_data = data[data["Environment"].str.contains("_")]

        raw_train, raw_test = train_test_split(raw_data, test_size=0.2, random_state=42)
        train_data = pd.concat([non_raw_data, raw_train], ignore_index=True)
        test_data = raw_test

        # Split train into actual train + val for tuning
        val_data, train_data = train_test_split(
            train_data, train_size=0.8, random_state=42, stratify=train_data["Environment"]
        )

        X_val = val_data.drop(columns=drop_columns)
        y_val = val_data[target_column]

        X_train = train_data.drop(columns=drop_columns)
        y_train = train_data[target_column]

        # Scale features (optional for RF, but keeps parity)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_val_scaled, y_val)
        best_model = grid.best_estimator_

        # Train on all training data
        X_all_train = pd.concat([X_train, X_val])
        y_all_train = pd.concat([y_train, y_val])
        X_all_scaled = scaler.fit_transform(X_all_train)

        best_model.fit(X_all_scaled, y_all_train)

        # Evaluate on test data
        X_test = test_data.drop(columns=drop_columns)
        y_test = test_data[target_column]
        X_test_scaled = scaler.transform(X_test)

        y_pred = best_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r = np.sqrt(r2) if r2 >= 0 else float("nan")

        result = permutation_importance(
            best_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean
        feature_importance = dict(zip(X_test.columns, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        results_df, results_path = self.save_results(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            r=r,
            top_features=top_features,
            model_type="randomforest"
        )

        return results_df, results_path
