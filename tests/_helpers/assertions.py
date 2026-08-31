"""Shared assertions used across unit / integration / regression layers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# The scalar metric keys EvaluatorSl.evaluate always produces.
SL_METRIC_KEYS = {
    "roc_auc",
    "bacc",
    "f1",
    "precision",
    "recall",
    "pr_auc",
    "mcc",
    "gmean",
    "specificity",
    "sensitivity",
    "npv",
}


def assert_valid_metrics(metrics: dict) -> None:
    """EvaluatorSl.evaluate output: every scalar metric present, numeric,
    and in a sane range; confusion matrix internally consistent."""
    for key in SL_METRIC_KEYS:
        assert key in metrics, f"missing metric {key!r}"
        score = metrics[key]["score"] if isinstance(metrics[key], dict) else metrics[key]
        assert isinstance(score, (int, float)) and not math.isnan(score), key
    for key in ("roc_auc", "bacc", "f1", "precision", "recall", "specificity", "npv"):
        score = metrics[key]["score"]
        assert -0.0001 <= score <= 1.0001, f"{key} out of [0, 1]: {score}"
    assert -1.0001 <= metrics["mcc"]["score"] <= 1.0001
    cm = metrics["confusion_matrix"]
    assert {"TP", "TN", "FP", "FN"} == set(cm)
    assert all(isinstance(v, int) and v >= 0 for v in cm.values())


def assert_metrics_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    assert isinstance(data, dict) and data, f"empty metrics json: {path}"
    return data


def assert_feature_matrix(df: pd.DataFrame, id_col: str = "id") -> None:
    """A built feature matrix: has the id column, is unique per id, all other
    columns numeric, no NaN."""
    assert id_col in df.columns, f"no {id_col!r} column"
    assert df[id_col].is_unique, f"duplicate {id_col} rows"
    feats = df.drop(columns=[id_col])
    assert not feats.empty, "matrix has no feature columns"
    assert feats.apply(lambda c: pd.api.types.is_numeric_dtype(c)).all(), (
        "non-numeric feature column"
    )
    assert not feats.isna().any().any(), "NaN in feature matrix"


def assert_patient_level(ids) -> None:
    """Every id looks like ``<vtool>|S<number>`` with nothing after -- i.e.
    a patient id, never a contig id (which carries a ``_<contig>`` suffix)."""
    import re

    pat = re.compile(r"^[^|]+\|S\d+$")
    bad = [i for i in ids if not pat.match(str(i))]
    assert not bad, f"non-patient-level ids leaked: {bad[:5]}"
