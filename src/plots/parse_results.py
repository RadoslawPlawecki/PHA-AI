"""
@author: Radosław Pławecki
"""

import pandas as pd


def parse_results(data_path):
    df = pd.read_csv(data_path)
    df_long = (
        df.melt(id_vars="metric", var_name="variable", value_name="value")
    )
    parsed = df_long["variable"].str.extract(
        r"^(loocv|rcv)_(geN|VIB|VS2)_(comp|host|func)$"
    )
    parsed.columns = ["method", "gtool", "modality"]
    df_long = pd.concat([df_long, parsed], axis=1)
    return df_long.dropna(subset=["method"])
