"""
@author: Radosław Pławecki
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from abc import ABC, abstractmethod


@dataclass
class CVResults:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    test_idx: np.ndarray
    importance_mean: np.ndarray
    importance_std: Optional[np.ndarray] = None
    fold_results: List[Dict] = field(default_factory=list)
    folds: Optional[np.ndarray] = None     
    repeats: Optional[np.ndarray] = None


class BaseValidator(ABC):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def run(self, model_wrapper, X, y) -> CVResults:
        pass


class LOOCVValidator(BaseValidator):
    def run(self, model_wrapper, X, y) -> CVResults:
        loo = LeaveOneOut()
        y_true, y_pred, y_prob, all_test_idx = [], [], [], []
        feat_importances = np.zeros(X.shape[1])
        it = tqdm(loo.split(X), total=len(y), desc="Single-Omic LOOCV", disable=not self.verbose)
        for train_idx, test_idx in it:
            model_wrapper.fit(X[train_idx], y[train_idx])
            y_pred.append(model_wrapper.predict(X[test_idx])[0])
            y_prob.append(model_wrapper.predict_proba(X[test_idx])[0, 1])
            y_true.append(y[test_idx][0])
            all_test_idx.append(test_idx[0])
            feat_importances += model_wrapper.feature_importances_
        return CVResults(
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_prob=np.array(y_prob),
            test_idx=np.array(all_test_idx),
            importance_mean=feat_importances / len(y)
        )


class RepeatedCVValidator(BaseValidator):
    def __init__(self, n_splits=5, n_repeats=5, random_state=42, verbose=True):
        super().__init__(verbose)
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits

    def run(self, model_wrapper, X, y) -> CVResults:
        y_true_all, y_pred_all, y_prob_all, test_idx_all = [], [], [], []
        folds_all, repeats_all = [], []
        perm_importance_folds = []
        fold_results = []
        it = tqdm(self.cv.split(X, y), total=self.cv.get_n_splits(X, y), desc="Single-Omic Repeated CV", disable=not self.verbose)
        for fold_id, (train_idx, test_idx) in enumerate(it):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model_wrapper.fit(X_train, y_train)
            y_pred = model_wrapper.predict(X_test)
            y_prob = model_wrapper.predict_proba(X_test)[:, 1]
            r = permutation_importance(model_wrapper.model_, X_test, y_test, n_repeats=5)
            n_samples = len(test_idx)
            current_repeat = fold_id // self.n_splits
            current_fold = fold_id % self.n_splits
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)
            test_idx_all.extend(test_idx)
            folds_all.extend([current_fold] * n_samples)     
            repeats_all.extend([current_repeat] * n_samples)
            perm_importance_folds.append(r.importances_mean)
            fold_results.append({
                "fold_id": fold_id,
                "y_true": y_test,
                "y_pred": y_pred,
            })
        perm_importance_matrix = np.vstack(perm_importance_folds)
        return CVResults(
            y_true=np.array(y_true_all),
            y_pred=np.array(y_pred_all),
            y_prob=np.array(y_prob_all),
            test_idx=np.array(test_idx_all),
            importance_mean=perm_importance_matrix.mean(axis=0),
            importance_std=perm_importance_matrix.std(axis=0),
            fold_results=fold_results,
            folds=np.array(folds_all),  
            repeats=np.array(repeats_all)
        )


class LateFusionRepeatedCVValidator(BaseValidator):
    def __init__(self, n_splits=5, n_repeats=10, random_state=42, verbose=True):
        super().__init__(verbose)
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits

    def run(self, fusion_model_wrapper, X_data: dict, y: np.ndarray) -> CVResults:
        y_true_all, y_pred_all, y_prob_all, test_idx_all = [], [], [], []
        folds_all, repeats_all = [], []
        modalities = list(X_data.keys())
        perm_importances_folds = {m: [] for m in modalities}
        any_values = next(iter(X_data.values()))["values"]
        total = self.cv.get_n_splits(y=y)
        it = tqdm(self.cv.split(any_values, y), total=total, desc="Multi-Omic Late Fusion Repeated CV", disable=not self.verbose)
        for fold_id, (train_idx, test_idx) in enumerate(it):
            fold_fusion_model = clone(fusion_model_wrapper)
            X_train_filtered = {}
            X_test_filtered = {}
            for m in modalities:
                vals = X_data[m]["values"]
                if hasattr(vals, "iloc"):
                    X_train_filtered[m] = vals.iloc[train_idx]
                    X_test_filtered[m] = vals.iloc[test_idx]
                else:
                    X_train_filtered[m] = vals[train_idx]
                    X_test_filtered[m] = vals[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_fusion_model.fit(X_train_filtered, y_train)
            y_true_all.extend(y_test)
            y_pred_all.extend(fold_fusion_model.predict(X_test_filtered))
            y_prob_all.extend(fold_fusion_model.predict_proba(X_test_filtered)[:, 1])
            test_idx_all.extend(test_idx)
            folds_all.extend([current_fold] * n_samples)    
            repeats_all.extend([current_repeat] * n_samples)
            for m in modalities:
                r = permutation_importance(
                    fold_fusion_model.models_dict[m].model_, 
                    X_test_filtered[m], 
                    y_test, 
                    n_repeats=5, 
                    random_state=42
                )
                perm_importances_folds[m].append(r.importances_mean)
        mean_importance_dict = {}
        std_importance_dict = {}
        for m in modalities:
            matrix = np.vstack(perm_importances_folds[m])
            mean_importance_dict[m] = matrix.mean(axis=0)
            std_importance_dict[m] = matrix.std(axis=0)
        return CVResults(
            y_true=np.array(y_true_all),
            y_pred=np.array(y_pred_all),
            y_prob=np.array(y_prob_all),
            test_idx=np.array(test_idx_all),
            importance_mean=mean_importance_dict,
            importance_std=std_importance_dict,
            folds=np.array(folds_all),   
            repeats=np.array(repeats_all)
        )


class LateFusionLOOCVValidator(BaseValidator):
    def run(self, fusion_model_wrapper, X_data: dict, y: np.ndarray) -> CVResults:
        loo = LeaveOneOut()
        y_true_all, y_pred_all, y_prob_all, test_idx_all = [], [], [], []
        modalities = list(X_data.keys())
        feature_importances = {m: np.zeros(X_data[m]["values"].shape[1]) for m in modalities}
        any_values = next(iter(X_data.values()))["values"]
        it = tqdm(loo.split(any_values), total=len(y), desc="Multi-Omic Late Fusion LOOCV", disable=not self.verbose)
        for train_idx, test_idx in it:
            fold_fusion_model = clone(fusion_model_wrapper)
            X_train_filtered = {}
            X_test_filtered = {}
            for m in modalities:
                vals = X_data[m]["values"]
                if hasattr(vals, "iloc"):
                    X_train_filtered[m] = vals.iloc[train_idx]
                    X_test_filtered[m] = vals.iloc[test_idx]
                else:
                    X_train_filtered[m] = vals[train_idx]
                    X_test_filtered[m] = vals[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_fusion_model.fit(X_train_filtered, y_train)
            y_true_all.append(y_test[0])
            y_pred_all.append(fold_fusion_model.predict(X_test_filtered)[0])
            y_prob_all.append(fold_fusion_model.predict_proba(X_test_filtered)[0, 1])
            test_idx_all.append(test_idx[0])
            for m in modalities:
                feature_importances[m] += fold_fusion_model.models_dict[m].feature_importances_
        mean_importance_dict = {m: feature_importances[m] / len(y) for m in modalities}
        return CVResults(
            y_true=np.array(y_true_all),
            y_pred=np.array(y_pred_all),
            y_prob=np.array(y_prob_all),
            test_idx=np.array(test_idx_all),
            importance_mean=mean_importance_dict,
            importance_std=None  
        )


class EarlyFusionLOOCVValidator(BaseValidator):
    def run(self, model_wrapper, X_data: dict, y: np.ndarray) -> CVResults:
        loo = LeaveOneOut()
        y_true_all, y_pred_all, y_prob_all, test_idx_all = [], [], [], []
        modalities = list(X_data.keys())
        X_fused = np.hstack([
            X_data[m]["values"].values if hasattr(X_data[m]["values"], "values") else X_data[m]["values"]
            for m in modalities
        ])
        flat_feat_importances = np.zeros(X_fused.shape[1])
        it = tqdm(loo.split(X_fused), total=len(y), desc="Multi-Omic Early Fusion LOOCV", disable=not self.verbose)
        for train_idx, test_idx in it:
            fold_model = clone(model_wrapper)
            X_train, X_test = X_fused[train_idx], X_fused[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_model.fit(X_train, y_train)
            y_true_all.append(y_test[0])
            y_pred_all.append(fold_model.predict(X_test)[0])
            y_prob_all.append(fold_model.predict_proba(X_test)[0, 1])
            test_idx_all.append(test_idx[0])
            flat_feat_importances += fold_model.feature_importances_
        mean_flat_importance = flat_feat_importances / len(y)
        mean_importance_dict = {}
        current_idx = 0
        for m in modalities:
            n_features = X_data[m]["values"].shape[1]
            mean_importance_dict[m] = mean_flat_importance[current_idx : current_idx + n_features]
            current_idx += n_features
        return CVResults(
            y_true=np.array(y_true_all),
            y_pred=np.array(y_pred_all),
            y_prob=np.array(y_prob_all),
            test_idx=np.array(test_idx_all),
            importance_mean=mean_importance_dict,
            importance_std=None
        )


class EarlyFusionRepeatedCVValidator(BaseValidator):
    def __init__(self, n_splits=5, n_repeats=5, random_state=42, verbose=True):
        super().__init__(verbose)
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits

    def run(self, model_wrapper, X_data: dict, y: np.ndarray) -> CVResults:
        y_true_all, y_pred_all, y_prob_all, test_idx_all = [], [], [], []
        folds_all, repeats_all = [], []
        perm_importance_folds = []
        modalities = list(X_data.keys())
        X_fused = np.hstack([
            X_data[m]["values"].values if hasattr(X_data[m]["values"], "values") else X_data[m]["values"]
            for m in modalities
        ])
        total_splits = self.cv.get_n_splits(X_fused, y)
        it = tqdm(self.cv.split(X_fused, y), total=total_splits, desc="Multi-Omic Early Fusion Repeated CV", disable=not self.verbose)
        for fold_id, (train_idx, test_idx) in enumerate(it):
            fold_model = clone(model_wrapper)
            X_train, X_test = X_fused[train_idx], X_fused[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_model.fit(X_train, y_train)
            y_true_all.extend(y_test)
            y_pred_all.extend(fold_model.predict(X_test))
            y_prob_all.extend(fold_model.predict_proba(X_test)[:, 1])
            test_idx_all.extend(test_idx)
            folds_all.extend([current_fold] * n_samples)   
            repeats_all.extend([current_repeat] * n_samples)
            r = permutation_importance(
                fold_model.model_ if hasattr(fold_model, "model_") else fold_model, 
                X_test, 
                y_test, 
                n_repeats=5, 
                random_state=42
            )
            perm_importance_folds.append(r.importances_mean)
        perm_importance_matrix = np.vstack(perm_importance_folds)
        flat_mean = perm_importance_matrix.mean(axis=0)
        flat_std = perm_importance_matrix.std(axis=0)
        mean_importance_dict = {}
        std_importance_dict = {}
        current_idx = 0
        for m in modalities:
            n_features = X_data[m]["values"].shape[1]
            mean_importance_dict[m] = flat_mean[current_idx : current_idx + n_features]
            std_importance_dict[m] = flat_std[current_idx : current_idx + n_features]
            current_idx += n_features
        return CVResults(
            y_true=np.array(y_true_all),
            y_pred=np.array(y_pred_all),
            y_prob=np.array(y_prob_all),
            test_idx=np.array(test_idx_all),
            importance_mean=mean_importance_dict,
            importance_std=std_importance_dict,
            folds=np.array(folds_all),     
            repeats=np.array(repeats_all)
        )
        