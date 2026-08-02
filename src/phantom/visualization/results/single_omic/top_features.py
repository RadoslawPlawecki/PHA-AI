"""
@author: Radosław Pławecki
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from project.src.plots.plots_formatting import use_latex


def find_feature_importance_file(base_dir, tool, omic, mode, model):
    target_folder = None
    if os.path.exists(base_dir):
        for folder in os.listdir(base_dir):
            if folder.startswith(f"run_{model}_{tool}_{omic}_"):
                target_folder = folder
                break
    if not target_folder:
        return None
    file_path = os.path.join(base_dir, target_folder, mode, "feature_importance.csv")
    return file_path if os.path.exists(file_path) else None


def generate_importance_plots(base_dir, model):
    use_latex()
    tools = ['geN', 'VIB', 'VS2']
    tools_map = {'geN': 'geNomad', 'VIB': 'VIBRANT', 'VS2': 'VirSorter2'}
    modes = ['loocv', 'rcv']
    modes_map = {'loocv': 'Leave-One-Out CV', 'rcv': 'Repeated Stratified 5-Fold CV'}
    omics = ['comp', 'host', 'func']
    omics_map = {'comp': 'Phage genus', 'host': 'Host-association score', 'func': 'Function ratios'}
    feature_name_map = {
        'structural_ratio': 'Structural',
        'packaging_ratio': 'Packaging',
        'unknown_ratio': 'Unknown function',
        'defense_accessory_ratio': 'Defense \& accessory',
        'replication_recombination_ratio': 'Replication \& recombination',
        'lysis_ratio': 'Lysis',
        'other_function_ratio': 'Other function',
        'transcription_regulation_ratio': 'Transcription \& regulation',
        'lysogeny_ratio': 'Lysogeny'
    }
    c0_color = '#34495e'  
    c1_color = '#27ae60' 
    eq_color = '#bdc3c7'  
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    for omic in omics:
        fig, axes = plt.subplots(3, 2, figsize=(15, 11))
        has_any_data = False
        for row_idx, tool in enumerate(tools):
            for col_idx, mode in enumerate(modes):
                ax = axes[row_idx, col_idx]
                file_path = find_feature_importance_file(base_dir, tool, omic, mode, model)
                if file_path:
                    has_any_data = True
                    df = pd.read_csv(file_path)
                    df_top = df.sort_values(by='importance', ascending=False).head(10)
                    df_top = df_top.iloc[::-1] 
                    df_top['display_name'] = df_top['feature'].apply(lambda x: feature_name_map.get(x, x))
                    colors = []
                    for _, r in df_top.iterrows():
                        if r['mean_class_0'] > r['mean_class_1']:
                            colors.append(c0_color)
                        elif r['mean_class_1'] > r['mean_class_0']:
                            colors.append(c1_color)
                        else:
                            colors.append(eq_color)
                    bars = ax.barh(df_top['display_name'], df_top['importance'], color=colors, height=0.65, edgecolor='none')
                    for bar in bars:
                        width = bar.get_width()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e0e0e0')
                    ax.set_axisbelow(True)
                    ax.tick_params(axis='both', labelsize=11, colors='#333333')
                else:
                    ax.text(0.5, 0.5, "Brak Danych / Pliku", ha='center', va='center', color='gray')
                    ax.axis('off')
                if row_idx == 0:
                    ax.set_title(modes_map[mode], fontsize=14, fontweight='bold', pad=15)
                if col_idx == 0:
                    ax.set_ylabel(tools_map[tool], fontsize=14, fontweight='bold', labelpad=15)
                if row_idx == 2 and file_path:
                    ax.set_xlabel("Feature importance score", fontsize=14, labelpad=15)
        if not has_any_data:
            plt.close(fig)
            continue
        legend_elements = [
            Patch(facecolor=c0_color, label='Higher mean value in control group'),
            Patch(facecolor=c1_color, label='Higher mean value in target group'),
            Patch(facecolor=eq_color, label='Equal mean values between groups')
        ]
        x_bbox = 0.5
        if omic == "func":
            x_bbox = 0.52
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(x_bbox, 0.94),
                   ncol=1, fontsize=12, frameon=True, facecolor='#fdfdfd', edgecolor='#eaeaea')
        plt.suptitle(f"Top features: {omics_map[omic]}", 
                     fontsize=18, fontweight='bold', x=x_bbox, y=0.98, color='#2c3e50')
        plt.tight_layout()
        fig.subplots_adjust(top=0.83, hspace=0.38, wspace=0.35)
        
        output_pdf = f"plots/results/single_omic/{model}/feature_importance_{omic}.pdf"
        plt.savefig(output_pdf, bbox_inches='tight', format='pdf', dpi=300)
        plt.show()
        plt.close()

if __name__ == "__main__":
    model = "rf"
    results_path = f"data/results/single_omic/{model}"
    generate_importance_plots(results_path, model=model)
