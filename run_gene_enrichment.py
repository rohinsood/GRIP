import os
import pandas as pd
from gseapy import enrichr
import mygene
import json

def run_gene_enrichment(feature_csv_path, config_path="config.json", top_n=20):
    # ---- Load config ----
    with open(config_path) as f:
        config = json.load(f)

    experiment_name = config.get("experiment_name", "experiment")
    species = config.get("species", "mouse")  # Accepts "human", "mouse", etc.
    gene_column = config.get("voted_feature_column", "Gene")
    enrichr_libraries = [
        "GO_Biological_Process_2023",
        "GO_Cellular_Component_2023",
        "GO_Molecular_Function_2023",
        "Reactome_2022",
        "ARCHS4_Tissues",
        "Rare_Diseases_AutoRIF_Gene_Lists",
        "KEGG_2019_Mouse",
        "WikiPathways_2024_Mouse",
        "MGI_Mammalian_Phenotype_Level_4_2024",
        "Mouse_Gene_Atlas"
    ]


    # ---- Load gene list ----
    df = pd.read_csv(feature_csv_path)
    genes = df[gene_column].tolist()
    symbols = list(ensembl_to_symbol(genes).values())
    print(genes)

    # ---- Enrichment ----
    for lib in enrichr_libraries:
        print(f"Running enrichment for: {lib}")
        enr = enrichr(
            gene_list=symbols,
            gene_sets=lib,
            organism=species,
            outdir=None,
        )
        

        if enr.results is not None and not enr.results.empty:
            top_results = enr.results.sort_values("Adjusted P-value").head(top_n)
            output_path = os.path.join("results/analysis", f"{experiment_name}_{lib}_top_enrichment.csv")
            top_results.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
        else:
            print(f"No enrichment found for: {lib}")

def ensembl_to_symbol(ensembl_ids):
    mg = mygene.MyGeneInfo()
    query_res = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='mouse')
    
    # Create dict mapping ensembl -> symbol (if found)
    ensembl_to_symbol_map = {}
    for entry in query_res:
        if 'notfound' not in entry and 'symbol' in entry:
            ensembl_to_symbol_map[entry['query']] = entry['symbol']
    
    return ensembl_to_symbol_map
