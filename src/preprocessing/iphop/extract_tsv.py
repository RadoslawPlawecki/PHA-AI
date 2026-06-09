"""
@author: Radosław Pławecki
"""

from pathlib import Path
import pandas as pd

in_root = Path("data/iphop")
out_root = Path("data/modalities/raw")

for gtool_dir in in_root.iterdir():
    if not gtool_dir.is_dir():
        continue

    for file in gtool_dir.rglob("Host_prediction_to_genus_m90.csv"):
        df = pd.read_csv(file, delimiter=',')
        parts = Path(file).parts
        out_path = out_root / Path(gtool_dir).parts[1] / Path(gtool_dir).parts[2] / f"{parts[3]}.tsv"
        df.to_csv(out_path, sep='\t', index=False)
