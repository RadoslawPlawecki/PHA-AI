"""
@author: Radosław Pławecki
"""

from ml.data.config import SingleOmicConfig
from ml.data.data_loader import DataLoader
from ml.data.preprocessor import NearZeroVarianceFilter
from ml.analytics.logger import Logger
from ml.analytics.fisher_analyzer import FisherAnalyzer
from ml.analytics.reporter import ReportFormatter
from ml.analytics.saver import ExperimentSaver
from ml.ml.models import (
    SingleOmicModel,
    get_rf_model,
    get_catboost_model,
    get_xgb_model,
)
from ml.ml.validators import (
    LOOCVValidator,
    RepeatedCVValidator,
    CVResults,
)
from ml.ml.evaluator import EvaluatorSl
import os
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime


class SingleOmicClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.saver = None
        self.values = None
        self.labels = None
        self.feature_names = None
        self.sample_ids = None
        self.model = None

    def run(self):
        self._setup()
        self._load()
        self._fisher_analysis()
        self._build_model()
        self._validate()

    def _setup(self):
        exp_dir, self.logger, self.saver = self._create_experiment()
        self.logger.info("=== SINGLE-OMIC ALLERGY CLASSIFIER ===")
        self.saver.save_metadata(vars(self.config))

    def _create_experiment(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.vtool is not None and self.config.modality is not None:
            exp_name = f"run_{self.config.model_type}_{self.config.vtool}_{self.config.modality}_{timestamp}"
        else:
            exp_name = f"run_{self.config.model_type}_{timestamp}"
        exp_dir = os.path.join(self.config.out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.setup_logger(
            log_dir=exp_dir,
            log_filename=f"{timestamp}.log"
        )
        saver = ExperimentSaver(exp_dir=exp_dir)
        return exp_dir, logger, saver

    def _load(self):
        loader = DataLoader(input_path=self.config.in_file, logger=self.logger)
        X, self.labels, self.sample_ids = loader.load()
        self.logger.info("Applying Near-Zero Variance Filter...")
        nzv = NearZeroVarianceFilter(logger=self.logger, threshold=4e-5)
        self.values, self.feature_names = nzv.fit_transform(X)

    def _fisher_analysis(self):
        if not self.config.run_fisher:
            return
        self.logger.info("Running Fisher's Exact Test...")
        fisher = FisherAnalyzer()
        significant = fisher.run(self.values, self.labels, self.feature_names)
        if not significant:
            self.logger.info("No Significant Features Found.")
            return
        for name, p in significant:
            self.logger.info(f"{name} | p={p:.4e}")

    def _build_model(self):
        factories = {
            "rf": get_rf_model,
            "xgb": get_xgb_model,
            "catboost": get_catboost_model,
        }
        self.model = factories[self.config.model_type](
            use_smote=self.config.use_smote
        )

    def _validate(self):
        if self.config.run_loocv:
            self._run_validation("loocv", LOOCVValidator(verbose=True))
        if self.config.run_repeated:
            self._run_validation("rcv", RepeatedCVValidator(verbose=True))

    def _run_validation(self, name, validator):
        self.logger.info(f"--- Starting Evaluation: {name.upper()} ---")
        results = validator.run(self.model, self.values, self.labels)
        metrics = EvaluatorSl.evaluate(
            results.y_true,
            results.y_pred,
            results.y_prob,
            results.test_idx
        )
        self._log_results(results, metrics)
        self._save_results(name, results, metrics)

    def _log_results(self, results, metrics):
        self.logger.info(ReportFormatter.format_metrics(metrics))
        self.logger.info(
            ReportFormatter.format_confusion_matrix(
                metrics["confusion_matrix"]
            )
        )
        self.logger.info(
            ReportFormatter.format_misclassified_samples(
                results.y_true,
                results.y_pred,
                results.test_idx,
                self.sample_ids
            )
        )
        self.logger.info(
            ReportFormatter.format_top_features(
                results.importance_mean,
                self.feature_names,
                self.values,
                self.labels
            )
        )

    def _save_results(self, name, results, metrics):
        mapped_ids = (
            self.sample_ids.iloc[results.test_idx].tolist()
            if results.test_idx is not None
            else None
        )
        self.saver.save_feature_importance(
            results.importance_mean,
            self.feature_names,
            self.values,
            self.labels,
            subfolder=name
        )
        self.saver.save_predictions(
            y_true=results.y_true,
            y_pred=results.y_pred,
            y_prob=results.y_prob,
            sample_ids=mapped_ids,
            folds=getattr(results, "folds", None),
            repeats=getattr(results, "repeats", None),
            subfolder=name
        )
        self.saver.save_metrics(metrics, subfolder=name)

    def _finish(self):
        self.logger.info("=== END EXPERIMENT ===")
