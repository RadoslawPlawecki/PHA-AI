"""
Shared search engine for feature-engineering optimization (cherry, phagcn,
phavip). Operates on raw-merged data rather than a canonical preprocessed
file, since phavip's search varies the coverage/identity thresholds its own
preprocessing applies (see _load_preprocessed). The CheckV mask is the one
thing every tool gets uniformly -- applied right after loading, before any
tool-specific preprocessing or search runs.

Building blocks reused by the three strategies in strategies/: loading a
tool's raw-merged file, applying a candidate config (apply_<tool>_config),
and running an Optuna study over its search space (search_<tool>). Not a
strategy itself -- see optimizer.py for the orchestrator.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import optuna
import pandas as pd

from phantom.classification.ml.models import get_catboost_model, get_rf_model, get_xgb_model
from phantom.classification.ml.optimizer import FeatureExtractionOptimizer
from phantom.classification.ml.validators import LOOCVValidator, RepeatedCVValidator
from phantom.cli.features import FeatureCollectionPrompts, FeatureExtractionPrompts, FeatureOptimizationPrompts
from phantom.config.features import FeatureConfigManager
from phantom.features.pipelines.tools.cherry import CherryFeaturePipeline
from phantom.features.pipelines.tools.phagcn import PhagcnFeaturePipeline
from phantom.features.pipelines.tools.phavip import PhavipFeaturePipeline
from phantom.features.pipelines.matrix import build_taxonomy_matrix
from phantom.features.pipelines.utils import apply_mask, load_file

MODEL_FACTORIES = {
    "catboost": get_catboost_model,
    "rf": get_rf_model,
    "xgb": get_xgb_model,
}

VALIDATORS = {
    "loocv": LOOCVValidator,
    "rcv": RepeatedCVValidator,
}

CHERRY_SEARCH_SPACE = {
    "binary": [True, False],
    "min_patients": list(range(1, 11)),
    "feature_col": ['ncbi_phylum', 'ncbi_class', 'ncbi_order',
                    'ncbi_family', 'ncbi_genus', 'ncbi_species']
}

PHAGCN_SEARCH_SPACE = {
    "binary": [True, False],
    "min_patients": list(range(1, 11))
}

CHERRY_DEFAULT_TRIALS = 150
PHAGCN_DEFAULT_TRIALS = 80
PHAVIP_DEFAULT_TRIALS = 150

DEFAULT_TRIALS = {
    "cherry": CHERRY_DEFAULT_TRIALS,
    "phagcn": PHAGCN_DEFAULT_TRIALS,
    "phavip": PHAVIP_DEFAULT_TRIALS,
}


@dataclass
class RunConfig:
    model: str = "catboost"
    validator: str = "loocv"
    metric: str = "mcc"
    smote: bool = False


def _select_raw_merged_file(tool: str, config_mgr: FeatureConfigManager | None = None) -> Path:
    config_mgr = config_mgr or FeatureConfigManager()
    existing_versions = config_mgr.get_existing_versions()
    if not existing_versions:
        raise FileNotFoundError("No feature versions found in config.toml.")
    version = FeatureCollectionPrompts.ask_existing_version(existing_versions)
    if not version:
        raise FileNotFoundError("No version selected.")
    raw_merged_dir = config_mgr.get_stage_dir(version, "raw_merged") / tool
    files = sorted(raw_merged_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No merged {tool} files found in {raw_merged_dir}")
    selected = FeatureOptimizationPrompts.ask_file_choice(tool, files)
    if not selected:
        raise FileNotFoundError("No file selected.")
    return selected


def _load_preprocessed(tool: str, config_mgr: FeatureConfigManager | None = None) -> tuple[Path, pd.DataFrame]:
    """
    Loads a tool's raw-merged file, applies the CheckV mask, then whatever
    preprocessing isn't itself part of the search space -- so every
    strategy starts from the same masked input.

    cherry/phagcn have no tunable preprocessing knobs, so preprocess() runs
    once here. phavip's min_coverage/min_pident ARE search params, so only
    Category is precomputed; preprocess() runs per-config in
    apply_phavip_config, on the masked data.
    """
    path = _select_raw_merged_file(tool, config_mgr)
    df = load_file(path)
    mask_path = FeatureExtractionPrompts().ask_mask_file(path)
    df = apply_mask(df, mask_path)
    if tool == "cherry":
        return path, CherryFeaturePipeline().preprocess(df)
    if tool == "phagcn":
        return path, PhagcnFeaturePipeline().preprocess(df)
    if tool == "phavip":
        print("[INFO] Precomputing Regex categories to accelerate Optuna trials...")
        df["Category"] = PhavipFeaturePipeline().categorize_annotations(df["Annotation"])
        return path, df
    raise ValueError(f"Unknown tool: {tool}")


def run_study(study_name: str, objective_callable, n_trials: int,
              model_name: str, validator_name: str, target_metric: str,
              use_smote: bool, file_path: Path | None = None, sampler=None,
              verbose: bool = True):
    model = MODEL_FACTORIES[model_name](use_smote=use_smote)
    validator = VALIDATORS[validator_name](verbose=False)
    optimizer = FeatureExtractionOptimizer(
        model=model,
        validator=validator,
        target_metric=target_metric
    )

    def wrapped_objective(trial):
        return objective_callable(trial, optimizer=optimizer)

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)
    if verbose:
        print(f"\n[INFO] Starting Optuna optimization for {study_name.upper()}...")
        print(f"[INFO] Pipeline: Model={model_name.upper()}, Validator={validator_name.upper()}, Target Metric={target_metric.upper()}, SMOTE={use_smote}")
    study.optimize(wrapped_objective, n_trials=n_trials)
    if verbose:
        print_optimization_results(
            study=study,
            study_name=study_name,
            target_metric=target_metric,
            file_path=file_path
        )
    return study


def print_optimization_results(study, study_name: str, target_metric: str, file_path: Path | None = None):
    print(f"\n=== {study_name.upper()} OPTIMIZATION FINISHED ===")
    if file_path is not None:
        print(f"Analyzed: {file_path.resolve()}")
    print(f"Best Trial Score ({target_metric.upper()}): {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(
        "[NOTE] This is an optimistic, exploratory estimate -- the same data "
        "both selected and scored this configuration. Run the nested mode "
        "for a deleakaged real-world estimate, the permutation mode for a "
        "significance check against chance."
    )


def apply_phagcn_config(
    df: pd.DataFrame, binary: bool, min_patients: int,
    feature_columns: list[str] | None = None
) -> pd.DataFrame:
    return build_taxonomy_matrix(
        df, feature_col="genus", min_patients=min_patients, binary=binary,
        feature_columns=feature_columns
    )


def phagcn_objective(trial, preprocessed_df: pd.DataFrame, optimizer: FeatureExtractionOptimizer,
                      y_override=None) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)
    feature_matrix = apply_phagcn_config(preprocessed_df, binary=binary_repr, min_patients=min_patients)
    return optimizer.run(feature_matrix, y_override=y_override)


def search_phagcn(preprocessed_df: pd.DataFrame, model_name: str, validator_name: str,
                   target_metric: str, use_smote: bool, n_trials: int | None = None,
                   y_override=None, file_path: Path | None = None, verbose: bool = True) -> optuna.Study:
    bound_objective = partial(phagcn_objective, preprocessed_df=preprocessed_df, y_override=y_override)
    grid_sampler = optuna.samplers.GridSampler(PHAGCN_SEARCH_SPACE)
    return run_study(
        study_name="phagcn_feature_optimization",
        objective_callable=bound_objective,
        n_trials=n_trials if n_trials is not None else PHAGCN_DEFAULT_TRIALS,
        model_name=model_name,
        validator_name=validator_name,
        target_metric=target_metric,
        use_smote=use_smote,
        file_path=file_path,
        sampler=grid_sampler,
        verbose=verbose,
    )


def apply_cherry_config(
    df: pd.DataFrame, feature_col: str, binary: bool, min_patients: int,
    feature_columns: list[str] | None = None
) -> pd.DataFrame:
    return build_taxonomy_matrix(
        df, feature_col=feature_col, min_patients=min_patients, binary=binary,
        feature_columns=feature_columns
    )


def cherry_objective(trial, preprocessed_df: pd.DataFrame, optimizer: FeatureExtractionOptimizer,
                      y_override=None) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)
    feature_col = trial.suggest_categorical("feature_col", [
        'ncbi_phylum', 'ncbi_class', 'ncbi_order',
        'ncbi_family', 'ncbi_genus', 'ncbi_species'
    ])
    feature_matrix = apply_cherry_config(
        preprocessed_df, feature_col=feature_col, binary=binary_repr, min_patients=min_patients
    )
    return optimizer.run(feature_matrix, y_override=y_override)


def search_cherry(preprocessed_df: pd.DataFrame, model_name: str, validator_name: str,
                   target_metric: str, use_smote: bool, n_trials: int | None = None,
                   y_override=None, file_path: Path | None = None, verbose: bool = True) -> optuna.Study:
    bound_objective = partial(cherry_objective, preprocessed_df=preprocessed_df, y_override=y_override)
    grid_sampler = optuna.samplers.GridSampler(CHERRY_SEARCH_SPACE)
    return run_study(
        study_name="cherry_feature_optimization",
        objective_callable=bound_objective,
        n_trials=n_trials if n_trials is not None else CHERRY_DEFAULT_TRIALS,
        model_name=model_name,
        validator_name=validator_name,
        target_metric=target_metric,
        use_smote=use_smote,
        file_path=file_path,
        sampler=grid_sampler,
        verbose=verbose,
    )


def apply_phavip_config(
    df: pd.DataFrame, min_coverage: float, min_pident: float,
    feature_columns: list[str] | None = None
) -> pd.DataFrame:
    pipeline = PhavipFeaturePipeline(min_coverage=min_coverage, min_pident=min_pident)
    preprocessed_df = pipeline.preprocess(df)
    if preprocessed_df.empty:
        return pd.DataFrame()
    return pipeline.build_feature_matrix(preprocessed_df, feature_columns=feature_columns)


def phavip_objective(trial, df: pd.DataFrame, optimizer: FeatureExtractionOptimizer,
                      y_override=None) -> float:
    min_coverage = trial.suggest_float("min_coverage", 0.5, 1.0)
    min_pident = trial.suggest_float("min_pident", 0.3, 1.0)
    feature_matrix = apply_phavip_config(df, min_coverage=min_coverage, min_pident=min_pident)
    if feature_matrix.empty or feature_matrix.shape[1] <= 1:
        return 0.0
    return optimizer.run(feature_matrix, y_override=y_override)


def search_phavip(df: pd.DataFrame, model_name: str, validator_name: str,
                   target_metric: str, use_smote: bool, n_trials: int | None = None,
                   y_override=None, file_path: Path | None = None, verbose: bool = True) -> optuna.Study:
    bound_objective = partial(phavip_objective, df=df, y_override=y_override)
    return run_study(
        study_name="phavip_feature_optimization",
        objective_callable=bound_objective,
        n_trials=n_trials if n_trials is not None else PHAVIP_DEFAULT_TRIALS,
        model_name=model_name,
        validator_name=validator_name,
        target_metric=target_metric,
        use_smote=use_smote,
        file_path=file_path,
        verbose=verbose,
    )


SEARCH_FNS = {
    "cherry": search_cherry,
    "phagcn": search_phagcn,
    "phavip": search_phavip,
}

APPLY_FNS = {
    "cherry": apply_cherry_config,
    "phagcn": apply_phagcn_config,
    "phavip": apply_phavip_config,
}
