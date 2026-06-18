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
from .config import SingleOmicConfig
import os
import pandas as pd
import argparse
from tqdm import tqdm


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
    config = SingleOmicConfig.from_args()
    logger = setup_logger('soac')
    logger.info("=== SINGLE-OMIC ALLERGY CLASSIFIER ===")
    loader = DataLoader(input_path=config.in_file, logger=logger)
    X, labels, sample_ids = loader.load()
    nzv_filter = NearZeroVarianceFilter(logger=logger, threshold=8e-5)
    values, feature_names = nzv_filter.fit_transform(X)
    if config.run_fisher:
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
    logger.info(f"Initializing {config.model_type.upper()}...")
    model = model_factories[config.model_type](use_smote=config.use_smote)
    if config.run_loocv:
        logger.info(f"--- SINGLE-OMIC LOOCV ---")
        results_loocv = LOOCVValidator(verbose=True).run(model, values, labels)
        metrics = log_experiment_results(results_loocv, feature_names, sample_ids, logger)
        CSVReporter.save_metrics(
            filepath=config.out_file, 
            experiment_name=f'loocv_{config.gtool}_{config.modality}', 
            metrics=metrics)
    if config.run_repeated:
        logger.info(f"--- SINGLE-OMIC REPEATED CV {config.model_type.upper()} ---")
        results_rep = RepeatedCVValidator(verbose=True).run(model, values, labels)
        metrics = log_experiment_results(results_rep, feature_names, sample_ids, logger)
        CSVReporter.save_metrics(
            filepath=config.out_file, 
            experiment_name=f'rcv_{config.gtool}_{config.modality}', 
            metrics=metrics)
    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
    main()
    