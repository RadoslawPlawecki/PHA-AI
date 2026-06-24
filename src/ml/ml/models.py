"""
@author: Radosław Pławecki
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


class SingleOmicModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, use_smote=False, smote_k=2, random_state=42):
        self.model = model
        self.use_smote = use_smote
        self.smote_k = smote_k
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        self.model_ = clone(self.model)
        if self.use_smote:
            k = min(self.smote_k, np.min(np.bincount(y)) - 1)
            if k > 0:
                smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X, y)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    @property
    def feature_importances_(self):
        if self.model_ is None:
            return np.zeros(1)
        if hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        elif hasattr(self.model_, "get_feature_importance"): 
            return self.model_.get_feature_importance()
        return np.zeros(1)


class MultiOmicModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models_dict=None, weights=None):
        self.models_dict = models_dict if models_dict is not None else {}
        self.weights = weights if weights is not None else {}

    def fit(self, X_blocks_dict, y):
        for modality, X in X_blocks_dict.items():
            if modality in self.models_dict:
                self.models_dict[modality].fit(X, y)
        return self

    def predict_proba(self, X_blocks_dict):
        probs = []
        modality_weights = []
        for modality, X in X_blocks_dict.items():
            if modality in self.models_dict:
                prob = self.models_dict[modality].predict_proba(X)[:, 1]
                probs.append(prob)
                weight = self.weights.get(modality, 1.0)
                modality_weights.append(weight)
        avg_prob = np.average(probs, axis=0, weights=modality_weights)
        return np.column_stack([1 - avg_prob, avg_prob])
    
    def predict(self, X_blocks_dict):
        return (self.predict_proba(X_blocks_dict)[:, 1] >= 0.5).astype(int)


def get_rf_model(use_smote=False, **params):
    default_params = {
        'n_estimators': 300,
        'max_depth': 3,
        'min_samples_split': 4,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'random_state': 42
    }
    default_params.update(params)
    return SingleOmicModel(RandomForestClassifier(**default_params), use_smote=use_smote)


def get_xgb_model(use_smote=False, **params):
    from xgboost import XGBClassifier
    default_params = {
        "n_estimators": 300,
        "max_depth": 2,          
        "learning_rate": 0.03,      
        "min_child_weight": 8,     
        "gamma": 1.0,          
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,           
        "objective": "binary:logistic",
        "eval_metric": "aucpr",     
        "random_state": 42
    }
    default_params.update(params)
    return SingleOmicModel(XGBClassifier(**default_params), use_smote=use_smote)


def get_catboost_model(use_smote=False, **params):
    from catboost import CatBoostClassifier
    default_params = {
        'iterations': 50,
        'depth': 3,
        'learning_rate': 0.05,
        'l2_leaf_reg': 5,
        'random_strength': 1,
        'bagging_temperature': 1,
        'auto_class_weights': 'Balanced',
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'verbose': 0,
        'random_state': 42
    }
    default_params.update(params)
    return SingleOmicModel(CatBoostClassifier(**default_params), use_smote=use_smote)
