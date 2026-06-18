"""
@author: Radosław Pławecki
"""


def plot_metric_run(metric_name="auc", method_name="loocv", model_name=None, savefig=None):
    import pandas as pd
    from .plot_metric import plot_metric
    data_path = f"data/ml/sml/SML_CB.csv"
    df = pd.read_csv(data_path)
    df_long = (
        df.melt(id_vars="metric", var_name="variable", value_name="value")
    )
    parsed = df_long["variable"].str.extract(
        r"^(loocv|rcv)_(geN|VIB|VS2)_(comp|host|func)$"
    )
    parsed.columns = ["method", "gtool", "modality"]
    df_long = pd.concat([df_long, parsed], axis=1)
    df_long = df_long.dropna(subset=["method"])
    plot_metric(df_long, metric_name, method_name, model_name, savefig)


models = ["rf", "catboost"]
metrics = ["auc", "balanced_accuracy", "f1", "precision", "recall", "specificity", "gmean", "nvp", "pr_auc", "mcc"]
methods = ["loocv", "rcv"]

for model in models:
    for method in methods:
        for metric in metrics:
            plot_metric_run(metric_name=metric, method_name=method, model_name=model, savefig=True)
