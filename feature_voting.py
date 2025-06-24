import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def vote_top_features(csv_paths, config_path="config.json", top_k=20):
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    experiment_name = config.get("experiment_name", "experiment")
    models_used = "_".join(config.get("models", []))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = f"results/feature_voting"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"fv_{experiment_name}_{models_used}_{timestamp}.csv")

    feature_importance = defaultdict(float)
    feature_frequency = defaultdict(float)

    dfs = [pd.read_csv(path) for path in csv_paths]
    all_df = pd.concat(dfs, ignore_index=True)

    rmse_values = all_df["rmse"].values
    r2_values = all_df["r2"].values
    rmse_min, rmse_max = rmse_values.min(), rmse_values.max()
    r2_min, r2_max = r2_values.min(), r2_values.max()

    all_df["rmse_scaled"] = (all_df["rmse"] - rmse_min) / (rmse_max - rmse_min)
    all_df["r2_scaled"] = (all_df["r2"] - r2_min) / (r2_max - r2_min)

    for _, row in all_df.iterrows():
        rmse_scaled = row["rmse_scaled"]
        r2_scaled = row["r2_scaled"]
        weight_adjustment = 1 + r2_scaled

        raw_features = row["top_features"].strip("; ")
        feature_dict = {
            pair.split(":")[0]: float(pair.split(":")[1])
            for pair in raw_features.split("; ")
            if ":" in pair
        }

        values = np.array(list(feature_dict.values()))
        if values.max() > values.min():
            feature_dict = {
                k: (v - values.min()) / (values.max() - values.min())
                for k, v in feature_dict.items()
            }

        for feature, score in feature_dict.items():
            if score > 0:
                feature_importance[feature] += score * weight_adjustment
                feature_frequency[feature] += (1 / (rmse_scaled + 1e-6)) * weight_adjustment

    for feature in feature_importance:
        feature_importance[feature] *= feature_frequency[feature]

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    df_sorted = pd.DataFrame(sorted_features, columns=["Gene", "Weight"])

    df_sorted.head(top_k).to_csv(output_path, index=False)
    print(f"Top {top_k} features saved to: {output_path}")

    return output_path
