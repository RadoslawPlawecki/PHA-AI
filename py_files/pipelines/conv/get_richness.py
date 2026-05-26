"""
@author: Radosław Pławecki
"""

import os
import pandas as pd

root = "data/phabox2/binary/cherry"
file_map = {
    "geN": "genomad",
    "VIB": "vibrant",
    "VS2": "virsorter2"
}
dfs = []
files = os.listdir(root)

for file in files:
    in_file = os.path.join(root, file)
    df = pd.read_csv(in_file)
    method_name = file_map[file[:3]]
    out = pd.DataFrame({
        "sample": df["id"].str.split("|").str[1],
        method_name: df.drop(columns="id").sum(axis=1)
    })
    dfs.append(out)


merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="sample", how="outer")

merged["label"] = (
    merged["sample"]
    .str.extract(r"S(\d+)")
    .astype(int)
    .le(34)
    .astype(int)
)

merged["sample_num"] = merged["sample"].str.extract(r"S(\d+)").astype(int)
merged = merged.sort_values("sample_num").drop(columns="sample_num")

merged.to_csv("data/phabox2/richness/CHR_R.csv", index=False)
