"""
@author: Radosław Pławecki
"""

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from imblearn.metrics import geometric_mean_score
import numpy as np


class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, y_prob):
        results = {}

        results["roc_auc"] = roc_auc_score(y_true, y_prob)
        results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        results["f1"] = f1_score(y_true, y_pred)

        results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred, zero_division=0)

        results["geometric_mean"] = geometric_mean_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results["sensitivity"] = results["recall"] 

        results["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

        results["confusion_matrix"] = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }

        return results
