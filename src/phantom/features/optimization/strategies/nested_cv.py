"""
Track B: nested CV. Exploratory search scores the winning config on the
same data that picked it, so that number runs optimistic. Here each outer
fold reruns the inner search on outer-train patients only, refits the
winner, and scores once on outer-test patients it never saw -- pooled
across folds this gives a deleakaged estimate.

Each fold also fits a second model on just the top-K most important
features (ablation), to check the signal isn't scattered noise.

Splits happen at the patient level; a test fold's matrix is always
reindexed to the train fold's column set (see matrix.py's feature_columns).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from phantom.classification.data.labeling import Labeling
from phantom.classification.data.preprocessor import NearZeroVarianceFilter
from phantom.features.pipelines.matrix import derive_patient_id, filter_accessions
from ..search_core import APPLY_FNS, MODEL_FACTORIES, SEARCH_FNS


@dataclass
class PooledResults:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    patient_ids: np.ndarray
    fold: np.ndarray
    repeat: np.ndarray


@dataclass
class NestedCVResult:
    full: PooledResults
    ablation: PooledResults
    per_fold_best_params: list = field(default_factory=list)
    per_fold_top_features: list = field(default_factory=list)
    skipped_folds: list = field(default_factory=list)


def _pool(rows: list[dict]) -> PooledResults:
    if not rows:
        empty = np.array([])
        return PooledResults(y_true=empty, y_pred=empty, y_prob=empty,
                              patient_ids=empty, fold=empty, repeat=empty)
    return PooledResults(
        y_true=np.concatenate([r["y_true"] for r in rows]),
        y_pred=np.concatenate([r["y_pred"] for r in rows]),
        y_prob=np.concatenate([r["y_prob"] for r in rows]),
        patient_ids=np.concatenate([r["patient_ids"] for r in rows]),
        fold=np.concatenate([r["fold"] for r in rows]),
        repeat=np.concatenate([r["repeat"] for r in rows]),
    )


def run_nested_cv(
    tool: str,
    df: pd.DataFrame,
    model_name: str,
    validator_name: str,
    target_metric: str,
    use_smote: bool,
    outer_folds: int = 5,
    outer_repeats: int = 3,
    inner_trials: int | None = None,
    top_k: int = 10,
    random_state: int = 42,
) -> NestedCVResult:
    search_fn = SEARCH_FNS[tool]
    apply_fn = APPLY_FNS[tool]

    df = filter_accessions(df)
    patient_ids_all = derive_patient_id(df["Accession"])
    unique_patients = pd.Index(patient_ids_all.unique())
    unique_labels = Labeling.derive_label(pd.Series(unique_patients))

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state
    )

    full_rows, ablation_rows = [], []
    per_fold_best_params, per_fold_top_features, skipped_folds = [], [], []

    total_folds = outer_folds * outer_repeats
    splits = enumerate(outer_cv.split(unique_patients, unique_labels))
    for fold_id, (train_pos, test_pos) in tqdm(splits, total=total_folds, desc=f"{tool.upper()} nested CV"):
        current_repeat = fold_id // outer_folds
        current_fold = fold_id % outer_folds
        train_patients = set(unique_patients[train_pos])
        test_patients = set(unique_patients[test_pos])

        tqdm.write(
            f"[INFO] {tool.upper()} nested CV -- repeat {current_repeat}, fold {current_fold} "
            f"({len(train_patients)} train / {len(test_patients)} test patients)"
        )

        train_rows = df[patient_ids_all.isin(train_patients)]
        test_rows = df[patient_ids_all.isin(test_patients)]

        study = search_fn(
            train_rows,
            model_name=model_name,
            validator_name=validator_name,
            target_metric=target_metric,
            use_smote=use_smote,
            n_trials=inner_trials,
            verbose=False,
        )
        best_params = study.best_params
        per_fold_best_params.append(best_params)

        X_train_df = apply_fn(train_rows, **best_params)
        if X_train_df.empty or X_train_df.shape[1] <= 1:
            skipped_folds.append({"fold_id": fold_id, "reason": "empty train feature matrix"})
            continue
        feature_columns = [c for c in X_train_df.columns if c != "id"]
        X_test_df = apply_fn(test_rows, **best_params, feature_columns=feature_columns)
        if X_test_df.empty:
            skipped_folds.append({"fold_id": fold_id, "reason": "empty test feature matrix"})
            continue

        y_train = Labeling.derive_label(X_train_df["id"])
        y_test = Labeling.derive_label(X_test_df["id"])
        if len(np.unique(y_train)) <= 1:
            skipped_folds.append({"fold_id": fold_id, "reason": "single-class train fold"})
            continue

        nzv = NearZeroVarianceFilter(threshold=4e-5)
        X_train_vals, feature_names = nzv.fit_transform(X_train_df[feature_columns])
        X_test_vals, _ = nzv.transform(X_test_df[feature_columns])
        if X_train_vals.shape[1] == 0:
            skipped_folds.append({"fold_id": fold_id, "reason": "no features survived NZV filter"})
            continue

        n_test = len(y_test)
        test_patient_ids = X_test_df["id"].to_numpy()

        model = MODEL_FACTORIES[model_name](use_smote=use_smote)
        model.fit(X_train_vals, y_train)
        full_rows.append({
            "y_true": y_test,
            "y_pred": model.predict(X_test_vals),
            "y_prob": model.predict_proba(X_test_vals)[:, 1],
            "patient_ids": test_patient_ids,
            "fold": np.full(n_test, current_fold),
            "repeat": np.full(n_test, current_repeat),
        })

        k = min(top_k, len(feature_names))
        top_idx = np.argsort(model.feature_importances_)[-k:][::-1]
        top_feature_names = [feature_names[i] for i in top_idx]
        per_fold_top_features.append(top_feature_names)

        mini_model = MODEL_FACTORIES[model_name](use_smote=use_smote)
        mini_model.fit(X_train_vals[:, top_idx], y_train)
        ablation_rows.append({
            "y_true": y_test,
            "y_pred": mini_model.predict(X_test_vals[:, top_idx]),
            "y_prob": mini_model.predict_proba(X_test_vals[:, top_idx])[:, 1],
            "patient_ids": test_patient_ids,
            "fold": np.full(n_test, current_fold),
            "repeat": np.full(n_test, current_repeat),
        })

    return NestedCVResult(
        full=_pool(full_rows),
        ablation=_pool(ablation_rows),
        per_fold_best_params=per_fold_best_params,
        per_fold_top_features=per_fold_top_features,
        skipped_folds=skipped_folds,
    )
