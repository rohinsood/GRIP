import os
import pandas as pd
from dowhy import CausalModel

def run_dowhy_analysis(data_path, target_column, voted_csv_path, output_dir):
    # Load full dataset
    df = pd.read_csv(data_path)

    # Get top genes from voted CSV
    voted_df = pd.read_csv(voted_csv_path)
    top_genes = voted_df["Gene"].tolist()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for gene in top_genes:
        print(f"[DoWhy] Analyzing gene: {gene}")

        model = CausalModel(
            data=df,
            treatment=gene,
            outcome=target_column,
            common_causes=[]  # Removed environment_column
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand)

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        print(f"Estimated ATE for {gene}: {estimate.value}")

        results.append({
            "gene": gene,
            "ate": estimate.value
        })

    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "dowhy_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"[DoWhy] Saved results to: {results_csv}")

    return results_csv
