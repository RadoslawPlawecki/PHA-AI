"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .feature_extractor import PhatypFeatureExtractor

extractor = PhatypFeatureExtractor()

in_root = Path("data/modalities/2.0/raw-merged/phatyp")
out_root = Path("data/modalities/2.0/features/phatyp")

for file in in_root.iterdir():
    final_df = extractor.process_file(
        in_root=file,
        out_root=out_root
    )
    print(final_df.head())
