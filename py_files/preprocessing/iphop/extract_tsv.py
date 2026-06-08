"""
@author: Radosław Pławecki
"""

from pathlib import Path
import pandas as pd

in_root = Path("data/iphop")

for gtool_dir in in_root.iterdir():
    if not gtool_dir.is_dir():
        continue

    for file in gtool_dir.rglob("Host_prediction_to_genus_m90.csv"):
        df = pd.read_csv(file, delimiter=',')
        print(df)