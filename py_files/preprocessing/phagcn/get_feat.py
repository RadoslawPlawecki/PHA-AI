"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .feature_extractor import PhagcnFeatureExtractor

extractor = PhagcnFeatureExtractor()

in_root = Path("data/modalities/raw-merged/phagcn")
out_root = Path("data/modalities/features/phagcn")

for file in in_root.iterdir():
    final_df = extractor.process_file(
        in_root=file,
        out_root=out_root
    )
    print(final_df.head())
