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
import numpy as np
from sklearn.metrics import silhouette_score
from skbio.stats.distance import DistanceMatrix, anosim, permanova


class EvaluatorSl:
    METRICS = {
        "roc_auc": lambda yt, yp, ypb: roc_auc_score(yt, ypb),
        "bacc": lambda yt, yp, ypb: balanced_accuracy_score(yt, yp),
        "f1": lambda yt, yp, ypb: f1_score(yt, yp),
        "precision": lambda yt, yp, ypb: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp, ypb: recall_score(yt, yp, zero_division=0),
        "pr_auc": lambda yt, yp, ypb: average_precision_score(yt, ypb),
        "mcc": lambda yt, yp, ypb: matthews_corrcoef(yt, yp),
        "gmean": lambda yt, yp, ypb: geometric_mean_score(yt, yp),
    }


    @staticmethod
    def evaluate(y_true, y_pred, y_prob, test_idx):
        results = {}
        for name, metric_fn in EvaluatorSl.METRICS.items():
            score = metric_fn(y_true, y_pred, y_prob)
            metric_result = {
                "score": score
            }
            results[name] = metric_result
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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
        if test_idx is not None:
            fp_local = np.where((y_true == 0) & (y_pred == 1))[0]
            fn_local = np.where((y_true == 1) & (y_pred == 0))[0]
            fp_idx = test_idx[fp_local].tolist()
            fn_idx = test_idx[fn_local].tolist()
            results["misclassified"] = {
                "fp": fp_idx,
                "fn": fn_idx,
            }
        else:
            results["misclassified"] = None
        return results


class EvaluatorUl:
    METRICS = {
        "silhouette": lambda dist_matrix, labels:
            silhouette_score(dist_matrix, labels, metric="precomputed"),
        "anosim": lambda dist_matrix, labels:
            anosim(DistanceMatrix(dist_matrix), labels, permutations=999),
        "permanova": lambda dist_matrix, labels:
            permanova(DistanceMatrix(dist_matrix), labels, permutations=999),
    }

    @staticmethod
    def evaluate(dist_matrix, labels):
        results = {}
        for name, metric_fn in EvaluatorUl.METRICS.items():
            result = metric_fn(dist_matrix, labels)
            if name == "silhouette":
                results[name] = {
                    "score": float(result)
                }
            else:
                results[name] = {
                    "statistic": float(result["test statistic"]),
                    "p_value": float(result["p-value"]),
                }
        return results
        