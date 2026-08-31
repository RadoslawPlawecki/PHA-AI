"""Deterministic stand-ins for the slow / interactive parts of the pipeline.

- ``FakeEstimator``  -- instant, seedless replacement for rf/xgb/catboost.
- ``FakeValidator``  -- one train-on-all / predict-on-all pass, no CV loop.
- ``patch_prompts``  -- swap every named ``*Prompts`` method for a canned answer.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FakeEstimator(BaseEstimator, ClassifierMixin):
    """Predicts from a fixed threshold on feature 0; probabilities are that
    feature clipped to [0, 1]. Exposes ``feature_importances_`` (descending),
    so top-K ablation code has something monotone to sort."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.linspace(1.0, 0.0, num=max(X.shape[1], 1))
        return self

    def predict(self, X):
        col0 = np.asarray(X, dtype=float)[:, 0]
        return (col0 > self.threshold).astype(int)

    def predict_proba(self, X):
        p1 = np.clip(np.asarray(X, dtype=float)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def get_feature_importance(self):  # catboost-style accessor
        return self.feature_importances_


class FakeValidator:
    """Drop-in for LOOCVValidator / RepeatedCVValidator in fast tests: fits
    the wrapper once on everything and predicts on everything. Returns a real
    ``CVResults`` so downstream EvaluatorSl.evaluate works unchanged."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, model_wrapper, X, y):
        from phantom.classification.ml.validators import CVResults

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        model_wrapper.fit(X, y)
        return CVResults(
            y_true=y.copy(),
            y_pred=np.asarray(model_wrapper.predict(X)),
            y_prob=np.asarray(model_wrapper.predict_proba(X))[:, 1],
            test_idx=np.arange(len(y)),
            importance_mean=np.asarray(
                getattr(model_wrapper, "feature_importances_", np.zeros(X.shape[1]))
            ),
        )


def patch_prompts(monkeypatch, prompt_cls, answers: dict) -> None:
    """For each ``name -> value`` in ``answers``, replace
    ``prompt_cls.name`` with a callable returning ``value``. If ``value`` is
    an iterator, successive calls return successive items (menu loops)."""
    for name, value in answers.items():
        if isinstance(value, Iterator):
            resolver = lambda *_a, _it=value, **_k: next(_it)  # noqa: E731
        else:
            resolver = lambda *_a, _v=value, **_k: _v  # noqa: E731
        monkeypatch.setattr(prompt_cls, name, staticmethod(resolver))
