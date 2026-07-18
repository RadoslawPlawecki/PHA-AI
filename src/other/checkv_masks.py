"""
@author: Radosław Pławecki
"""

import pandas as pd
import os

checkv_path = "./data/checkv/"
output_base = "./data/masks/"

for tool in os.listdir(checkv_path):
    tool_path = os.path.join(checkv_path, tool)
    if not os.path.isdir(tool_path):
        continue
    print(f"\nProcessing tool: {tool}")
    out_dir = os.path.join(output_base, tool)
    os.makedirs(out_dir, exist_ok=True)
    dfs = []
    for sample in os.listdir(tool_path):
        sample_path = os.path.join(tool_path, sample)
        if not os.path.isdir(sample_path):
            continue
        sample_id, tool_tag = sample.split("_")[0], sample.split("_")[1]
        tsv_path = os.path.join(sample_path, "quality_summary.tsv")
        if not os.path.exists(tsv_path):
            continue
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            usecols=["contig_id", "provirus", "completeness", "contamination"],
        )
        df["contig_id"] = (
            tool_tag + "|" + sample_id + "_" +
            df["contig_id"]
        )
        dfs.append(df)
    if not dfs:
        continue
    merged = pd.concat(dfs, ignore_index=True)
    mask_50 = (
        (merged["provirus"] == "No") &
        (merged["completeness"] >= 50) &
        (merged["contamination"] < 10)
    )

    mask_50_provirus = (
        (merged["completeness"] >= 50) &
        (merged["contamination"] < 10)
    )

    mask_75 = (
        (merged["provirus"] == "No") &
        (merged["completeness"] >= 75) &
        (merged["contamination"] < 10)
    )

    mask_75_provirus = (
        (merged["completeness"] >= 75) &
        (merged["contamination"] < 10)
    )

    merged.loc[mask_50].to_csv(
        os.path.join(out_dir, "MQ_vMAGs.tsv"),
        sep="\t",
        index=False,
    )

    merged.loc[mask_75].to_csv(
        os.path.join(out_dir, "HQ_vMAGs.tsv"),
        sep="\t",
        index=False,
    )

    merged.loc[mask_50_provirus].to_csv(
        os.path.join(out_dir, "MQ_provMAGs.tsv"),
        sep="\t",
        index=False,
    )

    merged.loc[mask_75_provirus].to_csv(
        os.path.join(out_dir, "HQ_provMAGs.tsv"),
        sep="\t",
        index=False,
    )
