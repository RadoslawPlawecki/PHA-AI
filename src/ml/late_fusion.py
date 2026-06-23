
from .utils.logger import setup_logger
from ml.data_loader import DataLoader
from ml.preprocessor import NearZeroVarianceFilter
from .models import MultiOmicModel, get_rf_model, get_catboost_model, get_xgb_model
from .validators import LateFusionLOOCVValidator, LateFusionRepeatedCVValidator, CVResults
from .utils.reporter import ReportFormatter
from .evaluator import EvaluatorSl
from .utils.csv_reporter import CSVReporter
import os
import pandas as pd
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Omic Allergy Classifier")

    parser.add_argument("--comp", type=str, required=True, help="Direct path to the virus composition modality")
    parser.add_argument("--func", type=str, required=True, help="Direct path to the functional modality")
    parser.add_argument("--host", type=str, required=True, help="Direct path to the phage-host relation modality")
    parser.add_argument("--out_file", type=str, default=None, help="Direct path to an output file")

    parser.add_argument("--run_loocv", action="store_true", help="Run Leave-One-Out CV")
    parser.add_argument("--run_repeated", action="store_true", help="Run Repeated Stratified CV")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE to training sets")

    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost"])

    return parser.parse_args()


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
    args = parse_args()
    logger = setup_logger('moac')
    logger.info("=== MULTI-OMIC ALLERGY CLASSIFIER ===")
    paths = {
        "comp": args.comp,
        "func": args.func,
        "host": args.host,
    }
    model_factories = {
        "rf": get_rf_model,
        "xgb": get_xgb_model,
        "catboost": get_catboost_model
    }
    custom_weights = {
        "comp": 0.5,      
        "func": 1.0,     
        "host": 0.4     
    }
    fusion_models = {}
    X_data = {}
    labels = None
    sample_ids = None
    for m in paths:
        loader = DataLoader(input_path=paths[m], logger=logger)
        X, labels, sample_ids = loader.load()
        nzv_filter = NearZeroVarianceFilter(logger=logger, threshold=4e-5)
        values, feature_names = nzv_filter.fit_transform(X)
        X_data[m] = {
            "values": values,
            "feature_names": feature_names
        }
        fusion_models[m] = model_factories[args.model_type](use_smote=args.use_smote)
    model = MultiOmicModel(models_dict=fusion_models, weights=custom_weights)
    feature_names_dict = {m: X_data[m]["feature_names"] for m in X_data}
    if args.run_loocv:
        logger.info(f"--- MULTI-OMIC LOOCV {args.model_type.upper()} ---")
        results = LateFusionLOOCVValidator(verbose=True).run(model, X_data, labels)
        log_experiment_results(results, feature_names_dict, sample_ids, logger)
    if args.run_repeated:
        logger.info(f"--- MULTI-OMIC REPEATED CV {args.model_type.upper()} ---")
        results = LateFusionRepeatedCVValidator(n_splits=5, n_repeats=5, verbose=True).run(model, X_data, labels)
        log_experiment_results(results, feature_names_dict, sample_ids, logger)
    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
   main()
    