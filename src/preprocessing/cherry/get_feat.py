"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .feature_extractor import CherryFeatureExtractor

extractor = CherryFeatureExtractor()

in_root = Path("data/modalities/raw-merged/cherry")
out_root = Path("data/modalities/features/cherry")

for file in in_root.iterdir():
    final_df = extractor.process_file(
        in_root=file,
        out_root=out_root
    )
    print(final_df.head())
