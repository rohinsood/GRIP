import json
import pandas as pd
from data_augmentation import apply_augmentations
import os
from feature_voting import vote_top_features
from datetime import datetime
from run_gene_enrichment import run_gene_enrichment
import json
import os
import importlib
from datetime import datetime
import pandas as pd
from data_augmentation import apply_augmentations
from run_dowhy import run_dowhy_analysis

model_results_paths = []

def load_model_class(model_name):
    name_map = {
        "ann": ("ann_regressor", "ANNRegressor"),
        "ridge": ("ridge_regressor", "RidgeRegressor"),
        "naivebayes": ("naivebayes_regressor", "NaiveBayesRegressor"),
        "naive_bayes": ("naivebayes_regressor", "NaiveBayesRegressor"),
        "randomforest": ("randomforest_regressor", "RandomforestRegressor"),
        "random_forest": ("randomforest_regressor", "RandomforestRegressor"),
        "rf": ("randomforest_regressor", "RandomforestRegressor"),
    }
    if model_name not in name_map:
        raise ValueError(f"Model '{model_name}' not recognized.")
    
    module_file, class_name = name_map[model_name]
    module = importlib.import_module(f"regressions.{module_file}")
    return getattr(module, class_name)

def run_model(model_name, data, config, experiment_folder_name):
    ModelClass = load_model_class(model_name)
    regressor = ModelClass(data, target_column=config["target_column"], config=config, experiment_folder_name=experiment_folder_name)
    results, results_path = regressor.train_and_evaluate()

    model_results_paths.append(results_path)

    print(f"[{model_name}] Training completed.")
    print(results)


def main():
    with open("config.json") as f:
        config = json.load(f)

    target_column = config["target_column"]
    dataset_path = config["dataset_path"]
    augmentations = config.get("feature_augmentations", [])
    models = config.get("models", ["ann"])
    feature_voting_enabled = config.get("feature_voting", False)
    top_k = config.get("top_k_features", 20)
    biomedical_analysis_enabled = config.get("biomedical_analysis_enabled", False)
    dowhy_enabled = config.get("dowhy_enabled", False)
    run_models = config.get("run_models", False)

    # Load & augment data
    data = pd.read_csv(dataset_path)
    augmented_data = apply_augmentations(data, augmentations, dataset_path)

    experiment_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder_name = f"{experiment_name}_{timestamp}"
    results_dir = os.path.join("results/model", experiment_folder_name)
    os.makedirs(results_dir, exist_ok=True)

    if run_models:
        for model_name in models:
            run_model(model_name, augmented_data, config, experiment_folder_name)

    voted_csv_path = None
    if feature_voting_enabled and model_results_paths:
        voted_csv_path = vote_top_features(model_results_paths, config_path="config.json", top_k=top_k)

    if biomedical_analysis_enabled:
        run_gene_enrichment(voted_csv_path, config_path="config.json", top_n=top_k)

    if dowhy_enabled:
        if not voted_csv_path:
            raise ValueError("Feature voting must be enabled to run DoWhy analysis.")
        run_dowhy_analysis(
            data_path=dataset_path,
            target_column=target_column,
            voted_csv_path=voted_csv_path,
            output_dir=os.path.join("results/dowhy", experiment_folder_name)
        )

if __name__ == "__main__":
    main()
