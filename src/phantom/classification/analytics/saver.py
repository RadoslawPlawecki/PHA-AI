"""
@author: Radosław Pławecki
"""

import csv
import os
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ExperimentSaver:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir

    def _get_target_dir(self, subfolder):
        if subfolder:
            target_dir = os.path.join(self.exp_dir, subfolder)
            os.makedirs(target_dir, exist_ok=True)
            return target_dir
        return self.exp_dir

    def save_metadata(self, config_vars: dict):
        path = os.path.join(self.exp_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(config_vars, f, indent=4, cls=NumpyEncoder)

    def save_metrics(self, metrics: dict, subfolder: str = ""):
        metrics.pop("misclassified", None)
        cm = metrics.pop("confusion_matrix", None)
        target_dir = self._get_target_dir(subfolder)
        metrics_path = os.path.join(target_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        if cm:
            cm_path = os.path.join(target_dir, "confusion_matrix.json")
            with open(cm_path, "w") as f:
                json.dump(cm, f, indent=4, cls=NumpyEncoder)

    def save_feature_importance(self, importances: np.ndarray, feature_names: list, X: np.ndarray, y: np.ndarray, subfolder=""):
        target_dir = self._get_target_dir(subfolder)
        path = os.path.join(target_dir, "feature_importance.csv")
        mean_0 = [np.mean(X[y == 0, i]) for i in range(X.shape[1])]
        mean_1 = [np.mean(X[y == 1, i]) for i in range(X.shape[1])]
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "mean_class_0": mean_0,
            "mean_class_1": mean_1
        }).sort_values(by="importance", ascending=False)
        df.to_csv(path, index=False)

    def save_predictions(self, y_true, y_pred, y_prob, sample_ids=None, folds=None, repeats=None, subfolder=""):
        target_dir = self._get_target_dir(subfolder)
        path = os.path.join(target_dir, "predictions.csv")
        data = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob_0": 1.0 - np.array(y_prob),
            "y_prob_1": y_prob
        }
        if sample_ids is not None: 
            data["sample_id"] = sample_ids
        if folds is not None: 
            data["fold"] = folds
        if repeats is not None: 
            data["repeat"] = repeats
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def save_unsupervised_metrics(self, all_metrics: dict, subfolder: str = ""):
        target_dir = self._get_target_dir(subfolder)
        path = os.path.join(target_dir, "unsupervised_metrics.csv")
        flattened_data = {}
        for modality, metrics in all_metrics.items():
            flattened_data[modality] = {
                "silhouette_score": metrics.get("silhouette", {}).get("score", np.nan),
                "permanova_F": metrics.get("permanova", {}).get("statistic", np.nan),
                "permanova_p": metrics.get("permanova", {}).get("p_value", np.nan)
            }
        df = pd.DataFrame(flattened_data).T 
        df.index = df.index.str.capitalize()
        df.index.name = "Modality"
        df.to_csv(path)
