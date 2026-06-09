"""
@author: Radosław Pławecki
"""


def plot_metric_run(metric_name="auc", gtool_name="geN", tool_name="CHR", method_name="loocv"):
    import pandas as pd
    from .plot_metric import plot_metric

    tool_name_dict = {
        "CHR": "cherry",
        "PGN": "phagcn"
    }

    data_path = f"data/results/sml/rf/{tool_name_dict[tool_name]}/{gtool_name}_{tool_name}_RF.csv"
    df = pd.read_csv(data_path)

    df_long = (
        df.melt(id_vars="metric", var_name="variable", value_name="value")
    )

    parsed = df_long["variable"].str.extract(
        r"^(loocv|rcv)_mP(\d+)_(.+)$"
    )

    parsed.columns = ["method", "mP", "level"]

    df_long = pd.concat([df_long, parsed], axis=1)

    df_long["min_patients"] = df_long["mP"]

    df_long = df_long.dropna(subset=["method"])

    plot_metric(df_long, metric_name, method_name)


plot_metric_run(metric_name="balanced_accuracy", tool_name="CHR", method_name="loocv")
