"""
Track C: permutation test. Each rep reruns the full search (same
objective/sampler) on patient-level shuffled labels, at the same trial
budget as the real search -- a max-statistic null, "best score achievable
by chance under this search budget." A naive single-model permutation test
would understate this, since a search over many configs inflates the best
achievable score even on noise.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from phantom.classification.data.labeling import Labeling
from phantom.classification.ml.significance import Significance
from phantom.features.pipelines.matrix import derive_patient_id, filter_accessions
from ..search_core import DEFAULT_TRIALS, SEARCH_FNS


@dataclass
class PermutationTestResult:
    tool: str
    observed_score: float
    null_scores: np.ndarray
    p_value: float
    baseline_score: float
    n_permutations: int
    permutation_trials: int
    per_permutation_best_params: list = field(default_factory=list)


def run_permutation_test(
    tool: str,
    df: pd.DataFrame,
    model_name: str,
    validator_name: str,
    target_metric: str,
    use_smote: bool,
    n_permutations: int = 50,
    permutation_trials: int | None = None,
    observed_score: float | None = None,
    random_state: int = 42,
) -> PermutationTestResult:
    search_fn = SEARCH_FNS[tool]
    permutation_trials = permutation_trials if permutation_trials is not None else DEFAULT_TRIALS[tool]

    df = filter_accessions(df)
    patient_ids_all = derive_patient_id(df["Accession"])
    unique_patients = pd.Index(patient_ids_all.unique())
    true_labels = Labeling.derive_label(pd.Series(unique_patients))

    if observed_score is None:
        observed_study = search_fn(
            df, model_name=model_name, validator_name=validator_name,
            target_metric=target_metric, use_smote=use_smote, verbose=False,
        )
        observed_score = observed_study.best_value

    baseline_score = Significance.majority_class_baseline(true_labels, target_metric)

    rng = np.random.default_rng(random_state)
    null_scores = []
    per_permutation_best_params = []
    for rep in range(n_permutations):
        print(f"\n[INFO] {tool.upper()} permutation test -- rep {rep + 1}/{n_permutations}")
        shuffled_labels = rng.permutation(true_labels)
        y_override = pd.Series(shuffled_labels, index=unique_patients)
        study = search_fn(
            df, model_name=model_name, validator_name=validator_name,
            target_metric=target_metric, use_smote=use_smote,
            n_trials=permutation_trials, y_override=y_override, verbose=False,
        )
        null_scores.append(study.best_value)
        per_permutation_best_params.append(study.best_params)

    null_scores = np.array(null_scores)
    p_value = Significance.permutation_pvalue(observed_score, null_scores)

    return PermutationTestResult(
        tool=tool,
        observed_score=observed_score,
        null_scores=null_scores,
        p_value=p_value,
        baseline_score=baseline_score,
        n_permutations=n_permutations,
        permutation_trials=permutation_trials,
        per_permutation_best_params=per_permutation_best_params,
    )
