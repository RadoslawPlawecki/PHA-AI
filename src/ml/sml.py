"""
@author: Radosław Pławecki
"""

from .utils.logger import setup_logger
from ml.data_loader import DataLoader
from ml.preprocessor import NearZeroVarianceFilter
from .fisher_analyzer import FisherAnalyzer
from .models import SingleOmicModel, get_rf_model, get_catboost_model, get_xgb_model
from .validators import LOOCVValidator, RepeatedCVValidator, CVResults
from .utils.reporter import ReportFormatter
from .evaluator import Evaluator
from .utils.csv_reporter import CSVReporter
import os
import pandas as pd
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Single-Omic Allergy Classifier")

    parser.add_argument("--in_file", type=str, required=True, help="Direct path to an input file")
    parser.add_argument("--out_file", type=str, default=None, help="Direct path to an output file")

    parser.add_argument("--run_fisher", action="store_true", help="Run Fisher's Exact Test")
    parser.add_argument("--run_loocv", action="store_true", help="Run Leave-One-Out CV")
    parser.add_argument("--run_repeated", action="store_true", help="Run Repeated Stratified CV")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE to training sets")

    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost"])

    return parser.parse_args()


def log_experiment_results(results: CVResults, feature_names: list, sample_ids: pd.Series, logger):
    metrics = Evaluator.evaluate(results.y_true, results.y_pred, results.y_prob, results.test_idx)
    logger.info(ReportFormatter.format_metrics(metrics))
    logger.info(ReportFormatter.format_confusion_matrix(metrics["confusion_matrix"]))
    logger.info(ReportFormatter.format_misclassified_samples(
        results.y_true, results.y_pred, results.test_idx, sample_ids
    ))
    logger.info(ReportFormatter.format_top_features(results.importance_mean, feature_names))
    return metrics


def main():
    args = parse_args()
    logger = setup_logger('soac')
    logger.info("=== SINGLE-OMIC ALLERGY CLASSIFIER ===")
    loader = DataLoader(input_path=args.in_file, logger=logger)
    X, labels, sample_ids = loader.load()
    nzv_filter = NearZeroVarianceFilter(logger=logger, threshold=8e-5)
    values, feature_names = nzv_filter.fit_transform(X)
    if args.run_fisher:
        logger.info("Running Fisher's Exact Test...")
        fisher = FisherAnalyzer()
        significant = fisher.run(values, labels, feature_names)
        if not significant:
            logger.info("No significant features found (after FDR).")
        else:
            logger.info("Significant features:")
            for name, p in significant:
                logger.info(f"{name} | p={p:.4e}")
    model_factories = {
        "rf": get_rf_model,
        "xgb": get_xgb_model,
        "catboost": get_catboost_model
    }
    logger.info(f"Initializing {args.model_type.upper()}...")
    model = model_factories[args.model_type](use_smote=args.use_smote)
    if args.run_loocv:
        logger.info(f"--- SINGLE-OMIC LOOCV i ---")
        results_loocv = LOOCVValidator(verbose=True).run(model, values, labels)
        metrics = log_experiment_results(results_loocv, feature_names, sample_ids, logger)
        # CSVReporter.save_metrics(filepath=args.out_file, experiment_name=f'loocv_mP{args.min_patients}_{args.column}', metrics=metrics)
    if args.run_repeated:
        logger.info(f"--- SINGLE-OMIC REPEATED CV {args.model_type.upper()} ---")
        results_rep = RepeatedCVValidator(verbose=True).run(model, values, labels)
        metrics = log_experiment_results(results_rep, feature_names, sample_ids, logger)
        # CSVReporter.save_metrics(filepath=args.out_file, experiment_name=f'rcv_mP{args.min_patients}_{args.column}', metrics=metrics)
    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
    main()
    