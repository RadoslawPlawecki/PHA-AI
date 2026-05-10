"""
@author: Radosław Pławecki
"""

from sklearn.utils import resample
import numpy as np

class BootstrapCI:
    @staticmethod
    def compute(
        y_true,
        y_pred,
        y_prob,
        metric_fn,
        n_bootstrap=2000,
        alpha=0.95,
        random_state=42
    ):
        rng = np.random.RandomState(random_state)

        scores = []

        n = len(y_true)

        for _ in range(n_bootstrap):
            indices = rng.choice(n, n, replace=True)

            y_true_bs = y_true[indices]
            y_pred_bs = y_pred[indices]
            y_prob_bs = y_prob[indices]

            if len(np.unique(y_true_bs)) < 2:
                continue

            score = metric_fn(
                y_true_bs,
                y_pred_bs,
                y_prob_bs
            )

            scores.append(score)

        scores = np.sort(scores)

        lower = np.percentile(scores, (1 - alpha) / 2 * 100)
        upper = np.percentile(scores, (1 + alpha) / 2 * 100)

        return {
            "mean": np.mean(scores),
            "lower": lower,
            "upper": upper
        }
        