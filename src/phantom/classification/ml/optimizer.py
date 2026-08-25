"""
@author: Radosław Pławecki
"""

import optuna
import os
import tempfile
import pandas as pd
import numpy as np
from typing import Callable, Any
from phantom.classification.data.data_loader import DataLoader
from phantom.classification.data.preprocessor import NearZeroVarianceFilter
from .models import (
    get_catboost_model,
    MultiOmicModel,
)
from .validators import (
    LOOCVValidator,
    LateFusionLOOCVValidator,
)
from .evaluator import EvaluatorSl


class LateFusionWeightOptimizer:
    def __init__(self, model_factories, config, paths, logger=None, n_trials=30, target_metric="mcc"):
        self.model_factories = model_factories
        self.config = config
        self.paths = paths
        self.logger = logger
        self.n_trials = n_trials
        self.target_metric = target_metric

    def optimize(self, X_data, y_aligned):
        def objective(trial):
            weights = {
                "comp": trial.suggest_float("weight_comp", 0.0, 1.0),
                "func": trial.suggest_float("weight_func", 0.0, 1.0),
                "host": trial.suggest_float("weight_host", 0.0, 1.0),
            }
            fusion_models = {
                m: self.model_factories[self.config.model_type](use_smote=self.config.use_smote) 
                for m in self.paths
            }
            model = MultiOmicModel(models_dict=fusion_models, weights=weights)
            validator = LateFusionLOOCVValidator(verbose=False)
            results = validator.run(model, X_data, y_aligned)
            metrics = EvaluatorSl.evaluate(results.y_true, results.y_pred, results.y_prob, results.test_idx)
            score = metrics[self.target_metric]
            return score["score"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        best_weights = {
            "comp": study.best_params["weight_comp"],
            "func": study.best_params["weight_func"],
            "host": study.best_params["weight_host"],
        }
        best_score = study.best_value
        self._log_optimize(best_weights=best_weights, best_score=best_score)
        return best_weights
        
    def _log_optimize(self, best_weights: dict, best_score: float) -> None:
        msg = (
            f"\nOptuna optimization results:\n"
            f"Best score: {best_score:.4f}\n"
            f"\nOptimized weights:\n"
            f"Comp: {best_weights['comp']:.4f}\n"
            f"Func: {best_weights['func']:.4f}\n"
            f"Host: {best_weights['host']:.4f}\n"
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


class FeatureExtractionOptimizer:
    def __init__(self, model: Any, validator: Any, target_metric: str = 'mcc', 
                 nzv_threshold: float = 4e-5, min_features: int = 3, logger=None):
        self.model = model
        self.validator = validator
        self.target_metric = target_metric
        self.nzv_threshold = nzv_threshold
        self.min_features = min_features
        self.logger = logger

    def run(self, feature_matrix: pd.DataFrame, y_override: pd.Series | None = None) -> float:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as tmp:
            feature_matrix.to_csv(tmp.name, sep=';', index=False)
            tmp_path = tmp.name
        try:
            loader = DataLoader(input_path=tmp_path, logger=self.logger)
            X, labels, sample_ids = loader.load()
            if y_override is not None:
                mapped = sample_ids.map(y_override)
                if mapped.isna().any():
                    raise ValueError(
                        "y_override does not cover every patient id in the feature matrix."
                    )
                labels = mapped.to_numpy(dtype=int)
            if X.empty or X.shape[1] < self.min_features or len(np.unique(labels)) <= 1:
                return 0.0
            nzv = NearZeroVarianceFilter(logger=self.logger, threshold=self.nzv_threshold)
            values, feature_names = nzv.fit_transform(X)
            if values.shape[1] == 0:
                return 0.0
            if isinstance(values, pd.DataFrame):
                values = values.copy()
            else:
                values = np.array(values, copy=True)
            results = self.validator.run(self.model, values, labels)
            metrics = EvaluatorSl.evaluate(
                results.y_true,
                results.y_pred,
                results.y_prob,
                results.test_idx
            )
            metric_value = metrics.get(self.target_metric, 0.0) 
            if isinstance(metric_value, dict):
                metric_value = metric_value.get('score', 0.0)   
            return float(metric_value)
        except KeyError:
            return 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                