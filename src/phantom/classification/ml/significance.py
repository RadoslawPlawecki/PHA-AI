"""
Significance-testing helpers for the permutation test
(features/optimization/strategies/permutation_search.py). No features.*
imports, mirroring the rest of this package (see evaluator.py).
"""

import numpy as np

from .evaluator import EvaluatorSl


class Significance:
    @staticmethod
    def permutation_pvalue(observed: float, null_scores: np.ndarray) -> float:
        null_scores = np.asarray(null_scores)
        return float((1 + np.sum(null_scores >= observed)) / (len(null_scores) + 1))

    @staticmethod
    def majority_class_baseline(y: np.ndarray, target_metric: str) -> float:
        """
        Score of a predictor that always outputs the majority class (and its
        implied constant probability) -- documents the near-chance level a
        target_metric sits at when a model learns nothing (MCC/gmean ~ 0).
        """
        y = np.asarray(y)
        majority = int(np.bincount(y).argmax())
        y_pred = np.full_like(y, fill_value=majority)
        y_prob = np.full(len(y), fill_value=float(majority))
        metrics = EvaluatorSl.evaluate(y, y_pred, y_prob, test_idx=None)
        metric_value = metrics.get(target_metric, 0.0)
        if isinstance(metric_value, dict):
            metric_value = metric_value.get("score", 0.0)
        return float(metric_value)
