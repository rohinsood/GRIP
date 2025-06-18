import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
import os

def check_and_replace_infs(df, replacement_value=1e6):
    df.replace([np.inf, -np.inf], replacement_value, inplace=True)
    df.fillna(replacement_value, inplace=True)
    return df

def transform_and_append(df, transformation, transformation_name, apply_to_whole_df=False):
    transformed_data = df.copy()

    if apply_to_whole_df:
        transformed_numeric = transformation(transformed_data.select_dtypes(include=[np.number]))
    else:
        transformed_numeric = transformed_data.select_dtypes(include=[np.number]).apply(transformation)

    for col in transformed_numeric.columns:
        transformed_numeric[col] = transformed_numeric[col].astype(transformed_data[col].dtype)

    transformed_numeric = check_and_replace_infs(transformed_numeric)

    transformed_data.update(transformed_numeric)
    transformed_data['Environment'] = transformed_data['Environment'] + f'_{transformation_name}'
    return transformed_data

def boxcox_transform(x):
    x_positive = x + 1 - np.min(x)
    transformed, _ = boxcox(x_positive)
    return transformed

def clr_transform(x):
    gm = np.exp(np.mean(np.log(x + 1)))  # Adding 1 to avoid log(0)
    return np.log((x + 1) / gm)  # Adding 1 to avoid log(0)

def mr_transform(df):
    transformed_data = df.copy()
    counts = transformed_data.select_dtypes(include=[np.number])
    geometric_means = np.exp(np.mean(np.log(counts + 1), axis=0))
    size_factors = counts.divide(geometric_means, axis=1).median(axis=0)
    normalized_counts = counts.divide(size_factors, axis=1)

    for col in normalized_counts.columns:
        normalized_counts[col] = normalized_counts[col].astype(transformed_data[col].dtype)

    normalized_counts = check_and_replace_infs(normalized_counts)
    transformed_data.update(normalized_counts)
    transformed_data['Environment'] = transformed_data['Environment'] + '_mr'
    return transformed_data

def zscore_transform(x):
    return (x - np.mean(x)) / np.std(x)

def quantile_transform(df):
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    transformed = qt.fit_transform(df.select_dtypes(include=[np.number]))
    transformed_df = pd.DataFrame(transformed, columns=df.select_dtypes(include=[np.number]).columns)
    transformed_df = check_and_replace_infs(transformed_df)
    return transformed_df

def minmax_transform(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def robust_transform(df):
    rs = RobustScaler()
    transformed = rs.fit_transform(df.select_dtypes(include=[np.number]))
    transformed_df = pd.DataFrame(transformed, columns=df.select_dtypes(include=[np.number]).columns)
    transformed_df = check_and_replace_infs(transformed_df)
    return transformed_df

def yeo_johnson_transform(df):
    pt = PowerTransformer(method='yeo-johnson')
    transformed = pt.fit_transform(df.select_dtypes(include=[np.number]))
    transformed_df = pd.DataFrame(transformed, columns=df.select_dtypes(include=[np.number]).columns)
    transformed_df = check_and_replace_infs(transformed_df)
    return transformed_df

# Mapping augmentation name to function
AUGMENTATION_FUNCTIONS = {
    "log": lambda df: transform_and_append(df, np.log1p, 'log'),
    "sqrt": lambda df: transform_and_append(df, np.sqrt, 'sqrt'),
    "boxcox": lambda df: transform_and_append(df, boxcox_transform, 'boxcox'),
    "clr": lambda df: transform_and_append(df, clr_transform, 'clr'),
    "mr": mr_transform,
    "zscore": lambda df: transform_and_append(df, zscore_transform, 'zscore'),
    "quantile": lambda df: transform_and_append(df, quantile_transform, 'quantile', apply_to_whole_df=True),
    "minmax": lambda df: transform_and_append(df, minmax_transform, 'minmax'),
    "robust": lambda df: transform_and_append(df, robust_transform, 'robust', apply_to_whole_df=True),
    "yeojohnson": lambda df: transform_and_append(df, yeo_johnson_transform, 'yeojohnson', apply_to_whole_df=True)
}

def apply_augmentations(df, augmentations, original_file_path):
    # Construct augmented filename based on augmentations
    base_filename = os.path.splitext(os.path.basename(original_file_path))[0]
    aug_suffix = "_" + "_".join(augmentations) if augmentations else ""
    output_filename = f"{base_filename}{aug_suffix}.csv"
    output_path = os.path.join("data", "ml_data", output_filename)

    # If file already exists, load and return it
    if os.path.exists(output_path):
        print(f"Augmented file found: {output_path}. Skipping augmentations.")
        return pd.read_csv(output_path)

    # Otherwise, perform augmentations
    print(f"No existing file for augmentations {augmentations}. Running augmentations...")
    augmented_dfs = [df]
    for aug in augmentations:
        if aug in AUGMENTATION_FUNCTIONS:
            augmented_df = AUGMENTATION_FUNCTIONS[aug](df)
            augmented_dfs.append(augmented_df)
        else:
            print(f"Warning: Augmentation '{aug}' not recognized and will be skipped.")

    combined = pd.concat(augmented_dfs, axis=0)
    combined.reset_index(drop=True, inplace=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Augmented data saved to {output_path}")

    return combined
