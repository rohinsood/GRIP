import json
import pandas as pd
from data_augmentation import apply_augmentations
from regressions.ann_regressor import ANNRegressor
import os
from datetime import datetime

def main():
    # Load config
    with open("config.json") as f:
        config = json.load(f)

    target_column = config["target_column"]
    n_features = config["n_features"]
    dataset_path = config["dataset_path"]
    use_invariance = config["use_invariance"]
    augmentations = config["feature_augmentations"]
    
    # Load data
    data = pd.read_csv(dataset_path)

    # Apply augmentations
    augmented_data = apply_augmentations(data, augmentations, dataset_path)

    experiment_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder_name = f"{experiment_name}_{timestamp}"

    results_dir = os.path.join("results", experiment_folder_name)
    os.makedirs(results_dir, exist_ok=True)


    # Initialize and train regressor
    ann = ANNRegressor(augmented_data, target_column=target_column, use_invariance=use_invariance, n_features=n_features)
    results = ann.train_and_evaluate()
      
    model_type = model.__class__.__name__.replace("Regressor", "").lower()
    inv_suffix = "_inv" if config.get("use_invariance") else ""
    result_filename = f"{timestamp}_{model_type}{inv_suffix}.csv"
    result_path = os.path.join(results_dir, result_filename)

    # Convert results dictionary to single-row DataFrame and save
    results_df = pd.DataFrame([results])
    results_df.to_csv(result_path, index=False)


    result_path = os.path.join(results_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print("Training completed. Results:")
    print(results)

if __name__ == "__main__":
    main()
