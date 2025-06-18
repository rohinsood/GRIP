from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .base_regressor import BaseRegressor

class ANNRegressor(BaseRegressor):

    def train_and_evaluate(self):
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

        # Prepare validation features and target for hyperparameter tuning
        X_val = val_data.drop(columns=drop_columns)
        y_val = val_data[target_column]

        # Scale validation features
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)

        param_grid = {
            "hidden_layer_sizes": [(64, 32, 16), (128, 64, 32), (32, 16, 8)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [800, 1200, 1600],
        }

        # Step 1: Use validation data for hyperparameter tuning
        mlp = MLPRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_val_scaled, y_val)
        best_params = grid_search.best_params_

        # Step 2: Sequential environment training with incremental validation
        environments = train_data["Environment"].unique()
        model = MLPRegressor(random_state=42, **best_params)

        for i, e_train in enumerate(environments):
            train_env_data = train_data[train_data["Environment"] == e_train]
            X_train_env = train_env_data.drop(columns=drop_columns)
            y_train_env = train_env_data[target_column]

            X_train_scaled = scaler.transform(X_train_env)

            # Incremental training (warm start enabled)
            if i == 0:
                # First fit
                model.fit(X_train_scaled, y_train_env)
            else:
                # Continue training on next environment
                model.partial_fit(X_train_scaled, y_train_env)

            # Validate on all other environments after this incremental training
            for e_val in environments:
                if e_val == e_train:
                    continue
                val_env_data = train_data[train_data["Environment"] == e_val]
                X_val_env = val_env_data.drop(columns=drop_columns)
                y_val_env = val_env_data[target_column]
                X_val_scaled = scaler.transform(X_val_env)

                y_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val_env, y_pred)
                print(f"Validation MSE after training on {e_train}, tested on {e_val}: {mse:.4f}")

        # Final evaluation on test set
        X_test = test_data.drop(columns=drop_columns)
        y_test = test_data[target_column]
        X_test_scaled = scaler.transform(X_test)

        y_pred_test = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        r = np.sqrt(r2) if r2 >= 0 else float("nan")

        # Permutation feature importance on test set
        result = permutation_importance(
            model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
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
            model_type="ann"
        )

        return results_df



