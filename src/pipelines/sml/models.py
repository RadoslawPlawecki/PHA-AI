"""
@author: Radosław Pławecki
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class BaseModel:
    def __init__(self, model, use_smote=False, smote_k=2, random_state=42):
        self.model = model
        self.use_smote = use_smote
        self.smote_k = smote_k
        self.random_state = random_state

    def fit(self, X, y):
        if self.use_smote:
            k = min(self.smote_k, np.min(np.bincount(y)) - 1)
            if k > 0:
                smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                X, y = smote.fit_resample(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_


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
    return BaseModel(RandomForestClassifier(**default_params), use_smote=use_smote)


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
    return BaseModel(XGBClassifier(**default_params), use_smote=use_smote)


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
    return BaseModel(CatBoostClassifier(**default_params), use_smote=use_smote)
