"""
@author: Radosław Pławecki
"""

import os
import json
import pandas as pd


def compile_single_omic_metrics_to_csv(base_dir, output_csv, model):
    tools = ['geN', 'VIB', 'VS2']
    tools_map = {'geN': 'geNomad', 'VIB': 'VIBRANT', 'VS2': 'VirSorter2'}
    omics = ['comp', 'host', 'func']
    omics_map = {'comp': 'Comp', 'host': 'Host', 'func': 'Func'}
    modes = ['loocv', 'rcv']
    metric_keys = [
        "roc_auc", "specificity", "sensitivity", "bacc", 
        "precision", "recall", "f1", "gmean", "npv", "mcc", 
    ]
    data_dict = {
        "Metric": [key for key in metric_keys]
    }
    for tool in tools:
        for omic in omics:
            target_folder = None
            if os.path.exists(base_dir):
                for folder in os.listdir(base_dir):
                    if folder.startswith(f"run_{model}_{tool}_{omic}_"):
                        target_folder = folder
                        break
            for mode in modes:
                col_name = f"{tool}_{omic}_{mode}"
                col_data = []
                if target_folder:
                    file_path = os.path.join(base_dir, target_folder, mode, "metrics.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            metrics_data = json.load(f)
                        for key in metric_keys:
                            col_data.append(metrics_data.get(key, {}).get("score", None))
                    else:
                        col_data = [None] * len(metric_keys)
                else:
                    col_data = [None] * len(metric_keys)
                data_dict[col_name] = col_data
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False, sep=';')


def compile_multi_omic_metrics_to_csv(base_dir, output_csv, model):
    tools = ['geN', 'VIB', 'VS2']
    tools_map = {'geN': 'geNomad', 'VIB': 'VIBRANT', 'VS2': 'VirSorter2'}
    fusions = ['early', 'late']
    fusions_map = {'early': 'early_fusion', 'late': 'late_fusion'}
    modes = ['loocv', 'rcv']
    metric_keys = [
        "roc_auc", "specificity", "sensitivity", "bacc", 
        "precision", "recall", "f1", "gmean", "npv", "mcc", 
    ]
    data_dict = {
        "Metric": [key for key in metric_keys]
    }
    for tool in tools:
        for fusion in fusions:
            target_folder = None
            if os.path.exists(base_dir):
                for folder in os.listdir(base_dir):
                    if folder.startswith(f"run_{fusion}_fusion_{tool}_"):
                        target_folder = folder
                        break
            for mode in modes:
                col_name = f"{tool}_{fusion}_{mode}"
                col_data = []
                if target_folder:
                    file_path = os.path.join(base_dir, target_folder, mode, "metrics.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            metrics_data = json.load(f)
                        for key in metric_keys:
                            col_data.append(metrics_data.get(key, {}).get("score", None))
                    else:
                        col_data = [None] * len(metric_keys)
                else:
                    col_data = [None] * len(metric_keys)
                data_dict[col_name] = col_data
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False, sep=';')

if __name__ == "__main__":
    model = "catboost"
    results_path = f"data/results/multi_omic/host=cherry/{model}"
    output_filename = os.path.join(results_path, "all_metrics.csv") 
    compile_multi_omic_metrics_to_csv(results_path, output_filename, model=model)
    