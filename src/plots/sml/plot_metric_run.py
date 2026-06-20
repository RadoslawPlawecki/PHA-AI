"""
@author: Radosław Pławecki
"""

import pandas as pd
from .plot_metric import plot_metrics
from ..parse_results import parse_results


def generate_all_metric_grids(data_path="data/ml/sml/SML_CB.csv", savefig=False):
    df = parse_results(data_path)
    models = ["rf", "catboost"]
    methods = ["loocv", "rcv"]
    for model in models:
        for method in methods:
            plot_metrics(
                df_long=df, 
                model_name="catboost", 
                method_name=method, 
                savefig=savefig
            )


generate_all_metric_grids()
