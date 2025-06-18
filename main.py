import json
import pandas as pd
from data_augmentation import apply_augmentations
from regressions.ann_regressor import ANNRegressor
import os
from datetime import datetime

import json
import os
import importlib
from datetime import datetime
import pandas as pd
from data_augmentation import apply_augmentations

def load_model_class(model_name):
    name_map = {
        "ann": ("ann_regressor", "ANNRegressor"),
        "ridge": ("ridge_regressor", "RidgeRegressor"),
        "naivebayes": ("naivebayes_regressor", "NaiveBayesRegressor"),
        "naive_bayes": ("naivebayes_regressor", "NaiveBayesRegressor"),
    }
    if model_name not in name_map:
        raise ValueError(f"Model '{model_name}' not recognized.")
    
    module_file, class_name = name_map[model_name]
    module = importlib.import_module(f"regressions.{module_file}")
    return getattr(module, class_name)

def run_model(model_name, data, config, experiment_folder_name, timestamp, results_dir):
    ModelClass = load_model_class(model_name)
    regressor = ModelClass(data, target_column=config["target_column"], config=config, experiment_folder_name=experiment_folder_name)
    results = regressor.train_and_evaluate()

    print(f"[{model_name}] Training completed.")
    print(results)

def main():
    with open("config.json") as f:
        config = json.load(f)

    target_column = config["target_column"]
    dataset_path = config["dataset_path"]
    augmentations = config.get("feature_augmentations", [])
    models = config.get("models", ["ann"])  # default to ANN if not specified

    # Load and augment data
    data = pd.read_csv(dataset_path)
    augmented_data = apply_augmentations(data, augmentations, dataset_path)

    experiment_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder_name = f"{experiment_name}_{timestamp}"
    results_dir = os.path.join("results", experiment_folder_name)
    os.makedirs(results_dir, exist_ok=True)

    # Loop through each model in config
    for model_name in models:
        run_model(model_name, augmented_data, config, experiment_folder_name, timestamp, results_dir)


if __name__ == "__main__":
    main()
