"""
@author: Radosław Pławecki
"""

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    matthews_corrcoef
)
from imblearn.metrics import geometric_mean_score
from .bookstrap_ci import BootstrapCI
import numpy as np


class Evaluator:
    METRICS = {
        "roc_auc": lambda yt, yp, ypb: roc_auc_score(yt, ypb),
        "balanced_accuracy": lambda yt, yp, ypb: balanced_accuracy_score(yt, yp),
        "f1": lambda yt, yp, ypb: f1_score(yt, yp),
        "precision": lambda yt, yp, ypb: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp, ypb: recall_score(yt, yp, zero_division=0),
        "pr_auc": lambda yt, yp, ypb: average_precision_score(yt, ypb),
        "mcc": lambda yt, yp, ypb: matthews_corrcoef(yt, yp),
        "geometric_mean": lambda yt, yp, ypb: geometric_mean_score(yt, yp),
    }

    CI_METRICS = {
        "roc_auc",
        "balanced_accuracy",
        "pr_auc",
    }


    @staticmethod
    def evaluate(y_true, y_pred, y_prob):
        results = {}

        for name, metric_fn in Evaluator.METRICS.items():

            score = metric_fn(y_true, y_pred, y_prob)

            metric_result = {
                "score": score
            }

            if name in Evaluator.CI_METRICS:
                metric_result["ci"] = BootstrapCI.compute(
                    y_true,
                    y_pred,
                    y_prob,
                    metric_fn
                )

            results[name] = metric_result

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        results["specificity"] = {
            "score": tn / (tn + fp) if (tn + fp) > 0 else 0.0
        }

        results["sensitivity"] = {
            "score": results["recall"]["score"]
        }

        results["npv"] = {
            "score": tn / (tn + fn) if (tn + fn) > 0 else 0.0
        }

        results["confusion_matrix"] = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }

        return results
