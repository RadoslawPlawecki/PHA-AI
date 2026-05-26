"""
@author: Radosław Pławecki
"""

from ..common.logger import setup_logger
from ..common.data_loader import DataLoader
from .analyze_fisher import FisherAnalyzer
from .models import *
from .validators import *
from .reporter import (
    format_metrics,
    format_confusion_matrix,
    format_top_features
)
from .evaluator import Evaluator
from .csv_reporter import CSVReporter
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Classifier")

    parser.add_argument("--in_file", type=str, required=True, help="Direct path to an input file")
    parser.add_argument("--out_file", type=str, default=None, help="Direct path to an output file")
    parser.add_argument("--tool", type=str, default=None, help="Type of tool the data comes from")
    parser.add_argument("--column", type=str, default=None, help="Column with data selected as features")

    parser.add_argument("--min_patients", type=int, default=2, help="Number of patients sharing the same viral cluster (VC).")

    parser.add_argument("--run_fisher", action="store_true", help="Run Fisher's Exact Test")
    parser.add_argument("--run_loocv", action="store_true", help="Run Leave-One-Out CV")
    parser.add_argument("--run_repeated", action="store_true", help="Run Repeated Stratified CV")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE to training sets")

    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost"])

    return parser.parse_args()


def log_experiment_results(results, feature_names, importance_key, logger):
    metrics = Evaluator.evaluate(
        results["y_true"],
        results["y_pred"],
        results["y_prob"]
    )
    logger.info(format_metrics(metrics))
    logger.info(format_confusion_matrix(metrics["confusion_matrix"]))
    logger.info(
        format_top_features(
            results[importance_key],
            feature_names
        )
    )
    return metrics


def main():
    args = parse_args()

    logger = setup_logger(f'{args.tool}_mP{args.min_patients}_{args.column}')

    logger.info("=== START EXPERIMENT ===")

    logger.info("Loading data...")
    loader = DataLoader(
        input_path=args.in_file, 
        min_patients=args.min_patients,
        tool=args.tool
    )
    X, y, feature_names = loader.process(col=args.column)
    
    logger.info(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")

    if args.run_fisher:
        # --- FISHER ---
        logger.info("Running Fisher's Exact Test...")
        fisher = FisherAnalyzer()

        significant = fisher.run(X, y, feature_names)

        if not significant:
            logger.info("No significant features found (after FDR).")
        else:
            logger.info("Significant features:")
            for name, p in significant:
                logger.info(f"{name} | p={p:.4e}")

    if args.model_type == "rf":
        logger.info("Initializing Random Forest...")
        model = get_rf_model(use_smote=args.use_smote)
    elif args.model_type == "xgb":
        logger.info("Initializing XGBoost...")
        model = get_xgb_model(use_smote=args.use_smote)
    elif args.model_type == "catboost":
        logger.info("Initializing CatBoost...")
        model = get_catboost_model(use_smote=args.use_smote)

    if args.run_loocv:
        logger.info(f"--- LOOCV {args.model_type.upper()} ---")
        results_loocv = LOOCVValidator(verbose=True).run(model, X, y)
        metrics = log_experiment_results(results_loocv, feature_names, "feature_importances", logger)
        CSVReporter.save_metrics(filepath=args.out_file, experiment_name=f'loocv_mP{args.min_patients}_{args.column}', metrics=metrics)

    if args.run_repeated:
        logger.info(f"--- REPEATED CV {args.model_type.upper()} ---")
        results_rep = RepeatedCVValidator(verbose=True).run(model, X, y)
        metrics = log_experiment_results(results_rep, feature_names, "perm_importances", logger)
        CSVReporter.save_metrics(filepath=args.out_file, experiment_name=f'rcv_mP{args.min_patients}_{args.column}', metrics=metrics)

    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
    main()