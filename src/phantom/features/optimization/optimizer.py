"""
Optuna-based hyperparameter search over feature-engineering choices
(which taxonomy column, min-patient cutoff, binary-vs-count, CheckV
coverage/identity thresholds) for cherry, phagcn, and phavip. Operates on
raw-merged data (pre-preprocessing), since some searches vary the very
thresholds preprocessing applies.
"""

import argparse
from functools import partial
from pathlib import Path

import optuna
import pandas as pd

from phantom.classification.ml.models import get_catboost_model, get_rf_model, get_xgb_model
from phantom.classification.ml.optimizer import FeatureExtractionOptimizer
from phantom.classification.ml.validators import LOOCVValidator, RepeatedCVValidator
from phantom.cli.features import FeatureCollectionPrompts, FeatureOptimizationPrompts
from phantom.config.features import FeatureConfigManager
from phantom.features.pipelines.tools.cherry import CherryFeaturePipeline
from phantom.features.pipelines.tools.phagcn import PhagcnFeaturePipeline
from phantom.features.pipelines.tools.phavip import PhavipFeaturePipeline
from phantom.features.pipelines.matrix import build_taxonomy_matrix
from phantom.features.pipelines.utils import load_file

MODEL_FACTORIES = {
    "catboost": get_catboost_model,
    "rf": get_rf_model,
    "xgb": get_xgb_model,
}

VALIDATORS = {
    "loocv": LOOCVValidator,
    "rcv": RepeatedCVValidator,
}


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


def run_study(study_name: str, objective_callable, n_trials: int,
              model_name: str, validator_name: str, target_metric: str,
              use_smote: bool, file_path: Path, sampler=None):
    model = MODEL_FACTORIES[model_name](use_smote=use_smote)
    validator = VALIDATORS[validator_name](verbose=False)
    optimizer = FeatureExtractionOptimizer(
        model=model,
        validator=validator,
        target_metric=target_metric
    )

    def wrapped_objective(trial):
        return objective_callable(trial, optimizer=optimizer)

    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)
    print(f"\n[INFO] Starting Optuna optimization for {study_name.upper()}...")
    print(f"[INFO] Pipeline: Model={model_name.upper()}, Validator={validator_name.upper()}, Target Metric={target_metric.upper()}, SMOTE={use_smote}")
    study.optimize(wrapped_objective, n_trials=n_trials)
    print_optimization_results(
        study=study,
        study_name=study_name,
        target_metric=target_metric,
        file_path=file_path
    )
    return study


def print_optimization_results(study, study_name: str, target_metric: str, file_path: Path):
    print(f"\n=== {study_name.upper()} OPTIMIZATION FINISHED ===")
    print(f"Analyzed: {file_path.resolve()}")
    print(f"Best Trial Score ({target_metric.upper()}): {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def phagcn_objective(trial, preprocessed_df: pd.DataFrame, optimizer: FeatureExtractionOptimizer) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)
    feature_matrix = build_taxonomy_matrix(
        preprocessed_df, feature_col="genus", min_patients=min_patients, binary=binary_repr
    )
    return optimizer.run(feature_matrix)


def run_phagcn(args, config_mgr: FeatureConfigManager | None = None):
    path = _select_raw_merged_file("phagcn", config_mgr)
    df = load_file(path)
    preprocessed_df = PhagcnFeaturePipeline().preprocess(df)
    bound_objective = partial(phagcn_objective, preprocessed_df=preprocessed_df)
    search_space = {
        "binary": [True, False],
        "min_patients": list(range(1, 11))
    }
    grid_sampler = optuna.samplers.GridSampler(search_space)
    run_study(
        study_name="phagcn_feature_optimization",
        objective_callable=bound_objective,
        n_trials=80,
        model_name=args.model,
        validator_name=args.validator,
        target_metric=args.metric,
        use_smote=args.smote,
        file_path=path,
        sampler=grid_sampler
    )


def cherry_objective(trial, preprocessed_df: pd.DataFrame, optimizer: FeatureExtractionOptimizer) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)
    feature_col = trial.suggest_categorical("feature_col", [
        'ncbi_phylum', 'ncbi_class', 'ncbi_order',
        'ncbi_family', 'ncbi_genus', 'ncbi_species'
    ])
    feature_matrix = build_taxonomy_matrix(
        preprocessed_df, feature_col=feature_col, min_patients=min_patients, binary=binary_repr
    )
    return optimizer.run(feature_matrix)


def run_cherry(args, config_mgr: FeatureConfigManager | None = None):
    path = _select_raw_merged_file("cherry", config_mgr)
    df = load_file(path)
    preprocessed_df = CherryFeaturePipeline().preprocess(df)
    bound_objective = partial(cherry_objective, preprocessed_df=preprocessed_df)
    search_space = {
        "binary": [True, False],
        "min_patients": list(range(1, 11)),
        "feature_col": ['ncbi_phylum', 'ncbi_class', 'ncbi_order',
                        'ncbi_family', 'ncbi_genus', 'ncbi_species']
    }
    grid_sampler = optuna.samplers.GridSampler(search_space)
    run_study(
        study_name="cherry_feature_optimization",
        objective_callable=bound_objective,
        n_trials=150,
        model_name=args.model,
        validator_name=args.validator,
        target_metric=args.metric,
        use_smote=args.smote,
        file_path=path,
        sampler=grid_sampler
    )


def phavip_objective(trial, df: pd.DataFrame, optimizer: FeatureExtractionOptimizer) -> float:
    min_coverage = trial.suggest_float("min_coverage", 0.5, 1.0)
    min_pident = trial.suggest_float("min_pident", 0.3, 1.0)
    pipeline = PhavipFeaturePipeline(min_coverage=min_coverage, min_pident=min_pident)
    preprocessed_df = pipeline.preprocess(df)
    if preprocessed_df.empty:
        return 0.0
    feature_matrix = pipeline.build_feature_matrix(preprocessed_df)
    if feature_matrix.empty or feature_matrix.shape[1] <= 1:
        return 0.0
    return optimizer.run(feature_matrix)


def run_phavip(args, config_mgr: FeatureConfigManager | None = None):
    path = _select_raw_merged_file("phavip", config_mgr)
    df = load_file(path)
    print("[INFO] Precomputing Regex categories to accelerate Optuna trials...")
    df["Category"] = PhavipFeaturePipeline().categorize_annotations(df["Annotation"])
    bound_objective = partial(phavip_objective, df=df)
    run_study(
        study_name="phavip_feature_optimization",
        objective_callable=bound_objective,
        n_trials=150,
        model_name=args.model,
        validator_name=args.validator,
        target_metric=args.metric,
        use_smote=args.smote,
        file_path=path
    )


TOOL_RUNNERS = {
    "cherry": run_cherry,
    "phagcn": run_phagcn,
    "phavip": run_phavip,
}


class FeatureOptimizer:
    """
    Interactive entry point used by FeatureController's "Optimize features"
    step. For full control over model/validator/metric/SMOTE, run this
    module directly instead (see the argparse block below).
    """

    def __init__(self, config_mgr: FeatureConfigManager | None = None,
                 model: str = "catboost", validator: str = "loocv",
                 metric: str = "mcc", smote: bool = False):
        self.config_mgr = config_mgr
        self.args = argparse.Namespace(model=model, validator=validator, metric=metric, smote=smote)

    def run(self) -> None:
        tool = FeatureOptimizationPrompts.ask_tool_choice(list(TOOL_RUNNERS))
        if not tool:
            return
        TOOL_RUNNERS[tool](self.args, config_mgr=self.config_mgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Extraction Optimization")

    parser.add_argument("--modality", type=str, required=True,
                        choices=["comp", "host", "func"],
                        help="Target modality to optimize.")
    parser.add_argument("--model", type=str, default="catboost",
                        choices=["catboost", "rf", "xgb"],
                        help="Machine learning model to use for evaluation.")
    parser.add_argument("--validator", type=str, default="loocv",
                        choices=["loocv", "rcv"],
                        help="Validation strategy.")
    parser.add_argument("--metric", type=str, default="mcc",
                        choices=["gmean", "mcc"],
                        help="Target metric to maximize.")
    parser.add_argument("--smote", action="store_true",
                        help="Include this flag to enable SMOTE for class balancing.")

    args = parser.parse_args()

    modality_to_tool = {"comp": "phagcn", "host": "cherry", "func": "phavip"}
    TOOL_RUNNERS[modality_to_tool[args.modality]](args)
