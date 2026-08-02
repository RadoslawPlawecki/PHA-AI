"""
@author: Radosław Pławecki
"""

import numpy as np


class DataAligner:
    @staticmethod
    def align(X_raw: dict) -> tuple[dict, np.ndarray, list]:
        modalities = list(X_raw.keys())
        common_samples = set(X_raw[modalities[0]]["ids"])
        for m in modalities[1:]:
            common_samples = common_samples.intersection(X_raw[m]["ids"])
        common_samples = sorted(list(common_samples))
        X_aligned = {}
        y_aligned = None
        for m in modalities:
            id_to_idx = {sid: idx for idx, sid in enumerate(X_raw[m]["ids"])}
            aligned_indices = [id_to_idx[sid] for sid in common_samples]
            raw_vals = X_raw[m]["values"]
            aligned_vals = raw_vals.iloc[aligned_indices] if hasattr(raw_vals, "iloc") else raw_vals[aligned_indices]
            X_aligned[m] = {
                "values": aligned_vals,
                "feature_names": X_raw[m].get("feature_names")
            }
            raw_labels = X_raw[m]["labels"]
            modality_labels = np.array([raw_labels[idx] for idx in aligned_indices])
            if y_aligned is None:
                y_aligned = modality_labels
            else:
                assert np.array_equal(y_aligned, modality_labels), f"Labels are not consistent for modality: {m}"
        return X_aligned, y_aligned, common_samples
