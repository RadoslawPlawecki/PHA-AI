"""
@author: Radosław Pławecki
"""

from pathlib import Path
import pandas as pd

in_root = Path("data/modalities/raw")
out_root = Path("data/modalities/raw-merged")

for ptool_dir in in_root.iterdir():
    if not ptool_dir.is_dir():
        continue

    for gtool_dir in ptool_dir.iterdir():
        if not gtool_dir.is_dir():
            continue

        dfs = []
        sample_name = None

        for file in gtool_dir.glob("*.tsv"):
            if sample_name is None:
                parts = file.stem.split('_')[1:]
                sample_name = "_".join(parts)

            df = pd.read_csv(file, sep="\t")
            dfs.append(df)

        if not dfs:
            continue

        merged = pd.concat(dfs, ignore_index=True)

        out_dir = out_root / ptool_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{sample_name}_M.csv"
        merged.to_csv(out_path, sep=';', index=False)

        print(f"\nPTool: {ptool_dir.name} | GTool: {gtool_dir.name}")
        print(merged.shape)
        print(merged.head())
