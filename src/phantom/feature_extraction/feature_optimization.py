"""
@author: Radosław Pławecki
"""

import argparse
from functools import partial
from pathlib import Path

import optuna
import pandas as pd

from phantom.cli.prompts import ModalityFileSelector
from phantom.classification.ml.models import get_catboost_model, get_rf_model, get_xgb_model
from phantom.classification.ml.optimizer import FeatureExtractionOptimizer
from phantom.classification.ml.validators import LOOCVValidator, RepeatedCVValidator
from phantom.feature_extraction import (
    CherryFeatureExtractor,
    PhagcnFeatureExtractor,
    PhavipFeatureExtractor,
)
from phantom.feature_extraction.features import build_matrix
from .utils import format_accession, load_file


model_factories = {
    "catboost": get_catboost_model,
    "rf": get_rf_model,
    "xgb": get_xgb_model,
}

validators = {
    "loocv": LOOCVValidator,
    "rcv": RepeatedCVValidator,
}


def run_study(study_name: str, objective_callable, n_trials: int, 
              model_name: str, validator_name: str, target_metric: str, 
              use_smote: bool, file_path: Path, sampler=None):
    model = model_factories[model_name](use_smote=use_smote)
    validator = validators[validator_name](verbose=False)
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


def phagcn_objective(trial, df: pd.DataFrame, optimizer: FeatureExtractionOptimizer) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)  
    extractor = PhagcnFeatureExtractor()
    df = extractor.preprocess(df)
    feature_matrix = build_matrix(
        df=df, feature_col="genus",
        binary=binary_repr, min_patients=min_patients
    )
    return optimizer.run(feature_matrix)


def run_phagcn(args):
    selector = ModalityFileSelector(modality="phagcn")
    path = selector.select()
    df = load_file(path)
    bound_objective = partial(phagcn_objective, df=df)
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


def cherry_objective(trial, df: pd.DataFrame, optimizer: FeatureExtractionOptimizer) -> float:
    binary_repr = trial.suggest_categorical("binary", [True, False])
    min_patients = trial.suggest_int("min_patients", 1, 10)  
    feature_col = trial.suggest_categorical("feature_col", [
        'ncbi_phylum', 'ncbi_class', 'ncbi_order', 
        'ncbi_family', 'ncbi_genus', 'ncbi_species'
    ])
    extractor = CherryFeatureExtractor()
    df = extractor.preprocess(df)
    feature_matrix = build_matrix(
        df=df, feature_col=feature_col,
        binary=binary_repr, min_patients=min_patients
    )
    return optimizer.run(feature_matrix)


def run_cherry(args):
    selector = ModalityFileSelector(modality="cherry")
    path = selector.select()
    df = load_file(path)
    bound_objective = partial(cherry_objective, df=df)
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
    extractor = PhavipFeatureExtractor(min_coverage=min_coverage, min_pident=min_pident)
    df = extractor.preprocess(df)
    if df.empty:
        return 0.0
    feature_matrix = extractor.calculate_category_ratios(df)
    if feature_matrix.empty or feature_matrix.shape[1] <= 1:
        return 0.0
    return optimizer.run(feature_matrix)


def run_phavip(args):
    selector = ModalityFileSelector(modality="phavip")
    path = selector.select()
    df = load_file(path, "Genome")
    print("[INFO] Precomputing Regex categories to accelerate Optuna trials...")
    extractor = PhavipFeatureExtractor()
    df["Category"] = extractor.categorize_annotations(df["Annotation"])
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

    if args.modality == "comp":
        run_phagcn(args)
    elif args.modality == "host":
        run_cherry(args)
    elif args.modality == "func":
        run_phavip(args)
    else:
        print(f"Error: Unknown modality '{args.modality}'.")
