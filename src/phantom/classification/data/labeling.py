"""
Single source of truth for the patient -> class-label rule used across the
classification stage (and, for nested/permutation evaluation, directly on
patient ids derived from contig-level Accession columns, without needing to
round-trip through DataLoader).
"""

import numpy as np
import pandas as pd


class Labeling:
    @staticmethod
    def derive_label(sample_ids: pd.Series) -> np.ndarray:
        return (
            pd.to_numeric(
                sample_ids.str.extract(r'\|S(\d+)')[0],
                errors='coerce'
            ) <= 34
        ).to_numpy(dtype=int)
