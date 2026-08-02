"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .feature_extractor import PhavipFeatureExtractor

extractor = PhavipFeatureExtractor()

in_root = Path("data/modalities/2.0/raw-merged/phavip")
out_root = Path("data/modalities/2.0/features/phavip")

for file in in_root.iterdir():
    final_df = extractor.process_file(
        in_root=file,
        out_root=out_root
    )
    print(final_df.head())
