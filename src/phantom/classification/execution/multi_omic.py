"""
@author: Radosław Pławecki
"""

from phantom.classification.data.config import MultiOmicConfig
from phantom.classification.data.data_loader import DataLoader
from phantom.classification.data.data_aligner import DataAligner
from phantom.classification.data.preprocessor import NearZeroVarianceFilter
from phantom.classification.analytics.logger import Logger
from phantom.classification.analytics.reporter import ReportFormatter
from phantom.classification.analytics.saver import ExperimentSaver
from phantom.classification.analytics.visualizer import Visualizer
from phantom.classification.ml.models import (
    MultiOmicModel,
    get_rf_model,
    get_catboost_model,
    get_xgb_model,
)
from phantom.classification.ml.optimizer import LateFusionWeightOptimizer
from phantom.classification.ml.validators import (
    EarlyFusionLOOCVValidator,
    EarlyFusionRepeatedCVValidator,
    LateFusionLOOCVValidator,
    LateFusionRepeatedCVValidator,
    CVResults,
)
from phantom.classification.ml.evaluator import EvaluatorSl
import os
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime


class MultiOmicClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.saver = None
        self.X_raw = {}
        self.X_data = None
        self.labels = None
        self.common_samples = None
        self.sample_ids = None
        self.feature_names_dict = {}
        self.model = None
        self.model_factories = {
            "rf": get_rf_model,
            "xgb": get_xgb_model,
            "catboost": get_catboost_model,
        }

    def run(self):
        self._setup()
        self._load()
        self._build_model()
        self._validate()
        self._finish()

    def _setup(self):
        exp_dir, self.logger, self.saver = self._create_experiment()
        fusion_name = {
            "early": "EARLY FUSION",
            "late": "LATE FUSION"
        }.get(self.config.fusion, None)
        if fusion_name is None:
            raise ValueError(f"Unsupported fusion strategy: {self.config.fusion}")
        self.logger.info(f"=== MULTI-OMIC ALLERGY CLASSIFIER ({fusion_name}) ===")
        self.saver.save_metadata(vars(self.config))

    def _create_experiment(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"run_{self.config.fusion}_fusion_{timestamp}"
        exp_dir = os.path.join(self.config.out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.setup_logger(
            log_dir=exp_dir,
            log_filename=f"{timestamp}.log"
        )
        saver = ExperimentSaver(exp_dir=exp_dir)
        return exp_dir, logger, saver

    def _load(self):
        paths = {
            "comp": self.config.comp,
            "func": self.config.func,
            "host": self.config.host,
        }
        for m in paths:
            loader = DataLoader(input_path=paths[m], logger=self.logger)
            X, labels, sample_ids = loader.load()
            nzv = NearZeroVarianceFilter(logger=self.logger, threshold=4e-5)
            values, feature_names = nzv.fit_transform(X)
            self.X_raw[m] = {
                "values": values,
                "feature_names": feature_names,
                "ids": list(sample_ids),
                "labels": list(labels)
            }
        self.X_data, self.labels, self.common_samples = DataAligner.align(self.X_raw)
        self.feature_names_dict = {
            m: self.X_data[m]["feature_names"] for m in self.X_data
        }
        self.sample_ids = sample_ids 
        self.logger.info("Datasets Aligned")

    def _build_model(self):
        if self.config.fusion == "early":
            self.model = self.model_factories[self.config.model_type](use_smote=self.config.use_smote)
            self.loocv = EarlyFusionLOOCVValidator(verbose=True)
            self.rcv = EarlyFusionRepeatedCVValidator(n_splits=5, n_repeats=5, verbose=True)
        elif self.config.fusion == "late":
            weights = self._get_weights()
            fusion_models = {
                m: self.model_factories[self.config.model_type](
                    use_smote=self.config.use_smote
                )
                for m in self.X_data
            }
            self.model = MultiOmicModel(models_dict=fusion_models, weights=weights)
            self.loocv = LateFusionLOOCVValidator(verbose=True)
            self.rcv = LateFusionRepeatedCVValidator(n_splits=5, n_repeats=5, verbose=True)
        else:
            raise ValueError(f"Unsupported fusion: {self.config.fusion}")

    def _get_weights(self):
        if not self.config.opt:
            self.logger.info("Using Default Late-Fusion Weights")
            return {
                "comp": 0.3132,
                "func": 0.8761,
                "host": 0.1119,
            }
        self.logger.info(
            f"Starting Optuna Optimization ({self.config.n_trials} Trials)..."
        )
        return self._optimize()

    def _optimize(self):
        optimizer = LateFusionWeightOptimizer(
            self.model_factories,
            self.config,
            {
                "comp": self.config.comp,
                "func": self.config.func,
                "host": self.config.host,
            },
            self.logger,
            n_trials=self.config.n_trials,
        )
        return optimizer.optimize(self.X_data, self.labels)

    def _validate(self):
        if self.config.run_loocv:
            self._run_validation("loocv", self.loocv)
        if self.config.run_repeated:
            self._run_validation("rcv", self.rcv)

    def _run_validation(self, name, validator):
        self.logger.info(f"--- MULTI-OMIC EVALUATION: {name.upper()} ---")
        results = validator.run(self.model, self.X_data, self.labels)
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
        for modality, importances in results.importance_mean.items():
            self.logger.info(
                f"\n--- Feature Importance: {modality.upper()} ---"
            )
            names = self.feature_names_dict[modality]
            self.logger.info(
                ReportFormatter.format_top_features(
                    importances,
                    names,
                    self.X_data[modality]["values"],
                    self.labels
                )
            )

    def _save_results(self, name, results, metrics):
        mapped_ids = (
            self.sample_ids.iloc[results.test_idx].tolist()
            if results.test_idx is not None else None
        )
        for modality, importances in results.importance_mean.items():
            self.saver.save_feature_importance(
                importances=importances,
                feature_names=self.feature_names_dict[modality],
                X=self.X_data[modality]["values"],
                y=self.labels,
                subfolder=os.path.join(name, modality),
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
        Visualizer.plot_roc_curve(
            results.y_true, results.y_prob,
            title=f"ROC Curve -- {self.config.model_type.upper()} {self.config.fusion.upper()} FUSION ({name.upper()})",
            save_path=os.path.join(self.saver.exp_dir, name, "roc_curve.pdf"),
        )

    def _finish(self):
        self.logger.info("=== END EXPERIMENT ===")
