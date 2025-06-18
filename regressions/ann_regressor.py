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
        best_mlp = grid_search.best_estimator_

        environments = data["Environment"].unique()

        if self.use_invariance:
            # K-fold style training between different environments
            for e_train in environments:
                for e_test in environments:
                    if e_train == e_test:
                        continue

                    train_data_env = train_data[train_data["Environment"] == e_train]
                    test_data_env = train_data[train_data["Environment"] == e_test]

                    if train_data_env.empty or test_data_env.empty:
                        continue

                    X_train = train_data_env.drop(columns=drop_columns)
                    y_train = train_data_env[target_column]

                    X_test_env = test_data_env.drop(columns=drop_columns)
                    y_test_env = test_data_env[target_column]

                    # Use same scaler fitted on validation data
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test_env)

                    best_mlp.fit(X_train_scaled, y_train)

        # Final evaluation on test set
        X_test = test_data.drop(columns=drop_columns)
        y_test = test_data[target_column]

        X_test_scaled = scaler.transform(X_test)
        y_pred = best_mlp.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r = np.sqrt(r2)

        # Permutation feature importance on test set
        result = permutation_importance(
            best_mlp, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
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
            selected_features=selected_features.tolist(),
            top_features=top_features,
        )


        return results

