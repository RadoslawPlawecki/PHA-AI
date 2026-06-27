"""
@author: Radosław Pławecki
"""

from project.src.plots.plots_formatting import use_latex
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extract_sample_number(sample_id):
    match = re.search(r'S(\d+)', str(sample_id))
    return int(match.group(1)) if match else 0

def process_predictions(base_dir, tool, omic, validation_mode, model):
    target_folder = None
    if os.path.exists(base_dir):
        for folder in os.listdir(base_dir):
            if folder.startswith(f"run_{model}_{tool}_{omic}_"):
                target_folder = folder
                break
    if not target_folder:
        return None
    file_path = os.path.join(base_dir, target_folder, validation_mode, "predictions.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df['clean_sample_id'] = df['sample_id'].apply(lambda x: x.split('|')[-1] if '|' in str(x) else x)
    df['sample_num'] = df['clean_sample_id'].apply(extract_sample_number)
    df['is_misclassified'] = (df['y_true'] != df['y_pred']).astype(int)
    if validation_mode == "loocv":
        summary = df.groupby(['sample_num', 'clean_sample_id'])['is_misclassified'].first().reset_index()
    else:
        summary = df.groupby(['sample_num', 'clean_sample_id'])['is_misclassified'].mean().reset_index()
    return summary

def build_matrix_for_mode(base_dir, validation_mode, model):
    tools = ['geN', 'VIB', 'VS2']
    omics = ['comp', 'host', 'func']
    omics_map = {'comp': 'Comp', 'host': 'Host', 'func': 'Func'}
    matrices_by_tool = {}
    for tool in tools:
        all_samples = set()
        omic_data_dict = {}
        for omic in omics:
            res = process_predictions(base_dir, tool, omic, validation_mode, model=model)
            if res is not None:
                omic_data_dict[omic] = res
                all_samples.update(res['clean_sample_id'].tolist())
        if not omic_data_dict:
            continue
        sorted_samples = sorted(list(all_samples), key=lambda x: extract_sample_number(x))
        matrix_data = []
        row_labels = []
        for omic in omics:
            if omic in omic_data_dict:
                res = omic_data_dict[omic]
                sample_to_val = dict(zip(res['clean_sample_id'], res['is_misclassified']))
                row = [sample_to_val.get(s, 0.0) for s in sorted_samples]
                matrix_data.append(row)
                row_labels.append(omics_map[omic])
            else:
                matrix_data.append([0.0] * len(sorted_samples))
                row_labels.append(f"{omics_map[omic]} (No Data)")
        matrices_by_tool[tool] = pd.DataFrame(matrix_data, index=row_labels, columns=sorted_samples)
    return matrices_by_tool

def generate_separated_heatmaps(base_dir, model):
    use_latex()
    tools = ['geN', 'VIB', 'VS2']
    tools_map = {'geN': 'geNomad', 'VIB': 'VIBRANT', 'VS2': 'VirSorter2'}
    modes = ['loocv', 'rcv']
    validation_map = {'loocv': 'Leave-One-Out CV', 'rcv': 'Repeated Stratified 5-Fold CV'}
    for mode in modes:
        matrices = build_matrix_for_mode(base_dir, mode, model=model)
        if not matrices:
            print(f"No data provided for mode: {mode.upper()}")
            continue
        fig, axes = plt.subplots(3, 1, figsize=(19, 10), sharex=True)
        if mode == "loocv":
            cmap = "YlGn"  
            cbar_label = "Misclassified (1 = Error, 0 = Correct)"
        else:
            cmap = "Greens"  
            cbar_label = "Error rate"
        for t_idx, tool in enumerate(tools):
            ax = axes[t_idx]
            if tool in matrices:
                df_plot = matrices[tool]
                im = sns.heatmap(df_plot, cmap=cmap, vmin=0, vmax=1.0, 
                            cbar=(t_idx == 0), 
                            cbar_kws={"label": cbar_label, "orientation": "horizontal", "pad": 0.1} if t_idx == 0 else None,
                            linewidths=0.7, linecolor="#f0f0f0", ax=ax)
                if t_idx == 0:
                    cbar = im.collections[0].colorbar
                    cbar.set_label(cbar_label, fontsize=14, fontweight='bold') # Większy tytuł
                    cbar.ax.tick_params(labelsize=12)
                ax.tick_params(axis='x', labelsize=10, labelrotation=90)
                ax.tick_params(axis='y', labelsize=12, labelrotation=0)
                ax.set_ylabel(tools_map[tool], fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"No data for tool: {tools_map[tool]}", ha='center', va='center')
                ax.axis('off')
        axes[-1].set_xlabel("Samples", fontsize=14, fontweight='bold', labelpad=10)
        plt.suptitle(f"Misclassified samples: {validation_map[mode]}", 
                     fontsize=21, fontweight='bold', y=0.93)
        plt.tight_layout()
        fig.subplots_adjust(top=0.88, wspace=0.2)
        output_filename = f"plots/results/single_omic/{model}/misclassified_{mode}.pdf"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='pdf')
        plt.show()
        plt.close()


if __name__ == "__main__":
    model = "rf"
    results_path = f"data/results/single_omic/{model}"
    generate_separated_heatmaps(results_path, model=model)
