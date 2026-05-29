"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .process_phavip import PhavipFeatureExtractor

extractor = PhavipFeatureExtractor()

in_root = Path("data/modalities/raw-merged/phavip")
out_root = Path("data/modalities/features/phavip")

for file in in_root.iterdir():
    final_df = extractor.process_file(
        in_root=file,
        out_root=out_root
    )
    print(final_df.head())
