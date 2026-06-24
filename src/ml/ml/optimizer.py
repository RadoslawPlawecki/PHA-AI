"""
@author: Radosław Pławecki
"""

import optuna
from ml.ml.models import MultiOmicModel
from ml.ml.validators import LateFusionLOOCVValidator
from ml.ml.evaluator import EvaluatorSl


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
