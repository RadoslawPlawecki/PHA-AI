"""
@author: Radosław Pławecki
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from tqdm import tqdm


class LOOCVValidator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def run(self, model_wrapper, X, y):
        loo = LeaveOneOut()
        y_true, y_pred, y_prob = [], [], []
        feat_importances = np.zeros(X.shape[1])

        it = tqdm(loo.split(X), total=len(y), desc="LOOCV", disable=not self.verbose)

        for train_idx, test_idx in it:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_wrapper.fit(X_train, y_train)

            y_pred.append(model_wrapper.predict(X_test)[0])
            y_prob.append(model_wrapper.predict_proba(X_test)[0, 1])
            y_true.append(y_test[0])
            feat_importances += model_wrapper.feature_importances_
        
        return {
            "y_true": np.array(y_true), "y_pred": np.array(y_pred),
            "y_prob": np.array(y_prob), "feature_importances": feat_importances / len(y)
        }


class RepeatedCVValidator:
    def __init__(self, n_splits=5, n_repeats=20, random_state=42, verbose=True):
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.verbose = verbose

    def run(self, X, y):
        y_true_all, y_pred_all, y_prob_all = [], [], []
        perm_importances = np.zeros(X.shape[1])

        total = self.cv.get_n_splits(X, y)
        it = tqdm(self.cv.split(X, y), total=total, desc="Repeated Stratified 5-Fold", disable=not self.verbose)

        for train_idx, test_idx in it:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_wrapper.fit(X_train, y_train)

            y_true_all.extend(y_test)
            y_pred_all.extend(model_wrapper.predict(X_test))
            y_prob_all.extend(model_wrapper.predict_proba(X_test)[:, 1])

            r = permutation_importance(model_wrapper.model, X_test, y_test, n_repeats=5)
            perm_importances += r.importances_mean
            
        return {
            "y_true": np.array(y_true_all), "y_pred": np.array(y_pred_all),
            "y_prob": np.array(y_prob_all), "perm_importances": perm_importances / total
        }
