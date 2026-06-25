"""
@author: Radosław Pławecki
"""

from ml.ingestion.config import MultiOmicConfig
from ml.ingestion.data_aligner import DataAligner
from ml.ingestion.data_loader import DataLoader
from ml.ingestion.preprocessor import NearZeroVarianceFilter
from ml.analytics.logger import Logger
from ml.analytics.reporter import ReportFormatter
from ml.analytics.csv_reporter import CSVReporter
from ml.ml.models import (
    MultiOmicModel,
    get_rf_model,
    get_catboost_model,
    get_xgb_model,
)
from ml.ml.validators import (
    EarlyFusionLOOCVValidator,
    EarlyFusionRepeatedCVValidator,
    LateFusionLOOCVValidator,
    LateFusionRepeatedCVValidator,
    CVResults,
)
from ml.ml.evaluator import EvaluatorSl
from ml.ml.optimizer import LateFusionWeightOptimizer
import os
import pandas as pd
import argparse
from tqdm import tqdm


def log_experiment_results(results: CVResults, feature_names_dict: dict, sample_ids: pd.Series, logger):
    metrics = EvaluatorSl.evaluate(results.y_true, results.y_pred, results.y_prob, results.test_idx)
    logger.info(ReportFormatter.format_metrics(metrics))
    logger.info(ReportFormatter.format_confusion_matrix(metrics["confusion_matrix"]))
    logger.info(ReportFormatter.format_misclassified_samples(
        results.y_true, results.y_pred, results.test_idx, sample_ids
    ))
    for modality, importances in results.importance_mean.items():
        logger.info(f"\n--- Feature Importance for Modality: {modality.upper()} ---")
        names = feature_names_dict[modality]
        logger.info(ReportFormatter.format_top_features(importances, names))
    return metrics


def main():
    config = MultiOmicConfig.from_args()
    logger = Logger.setup_logger('moac')
    fusion_dict = {
        "early": "Early Fusion",
        "late": "Late Fusion"
    }
    if config.fusion in fusion_dict:
        logger.info(f"=== MULTI-OMIC ALLERGY CLASSIFIER ({fusion_dict[config.fusion].upper()}) ===")
    else:
        raise ValueError(f"Unsupported fusion strategy specified: {config.fusion}")
    paths = {
        "comp": config.comp,
        "func": config.func,
        "host": config.host,
    }
    model_factories = {
        "rf": get_rf_model,
        "xgb": get_xgb_model,
        "catboost": get_catboost_model
    }
    custom_weights = {
        "comp": 0.3132,      
        "func": 0.8761,     
        "host": 0.1119     
    }
    fusion_models = {}
    X_raw = {}
    for m in paths:
        loader = DataLoader(input_path=paths[m], logger=logger)
        X, labels, sample_ids = loader.load()
        nzv_filter = NearZeroVarianceFilter(logger=logger, threshold=4e-5)
        values, feature_names = nzv_filter.fit_transform(X)
        X_raw[m] = {
            "values": values,
            "feature_names": feature_names,
            "ids": list(sample_ids),
            "labels": list(labels)
        }
    X_data, y_aligned, common_samples = DataAligner.align(X_raw)
    logger.info(f"Dataset aligned")
    feature_names_dict = {m: X_data[m]["feature_names"] for m in X_data}
    if config.fusion == "early":
        model = model_factories[config.model_type](use_smote=config.use_smote)
        loocv = EarlyFusionLOOCVValidator(verbose=True)
        rcv = EarlyFusionRepeatedCVValidator(n_splits=5, n_repeats=5, verbose=True)
    else:
        logger.info(f"Starting Optuna optimization for modality weights ({config.n_trials} trials)...")
        optimizer = LateFusionWeightOptimizer(model_factories, config, paths, logger, n_trials=config.n_trials)
        optimized_weights = optimizer.optimize(X_data, y_aligned)
        fusion_models = {}
        for m in paths:
            fusion_models[m] = model_factories[config.model_type](use_smote=config.use_smote)
        model = MultiOmicModel(models_dict=fusion_models, weights=optimized_weights)
        loocv = LateFusionLOOCVValidator(verbose=True)
        rcv = LateFusionRepeatedCVValidator(n_splits=5, n_repeats=5, verbose=True)
    if config.run_loocv:
        logger.info(f"--- MULTI-OMIC LOOCV {config.model_type.upper()} ({config.fusion.upper()}) ---")
        results = loocv.run(model, X_data, labels)
        log_experiment_results(results, feature_names_dict, sample_ids, logger)
    if config.run_repeated:
        logger.info(f"--- MULTI-OMIC REPEATED CV {config.model_type.upper()} ({config.fusion.upper()}) ---")
        results = rcv.run(model, X_data, labels)
        log_experiment_results(results, feature_names_dict, sample_ids, logger)
    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
   main()
    