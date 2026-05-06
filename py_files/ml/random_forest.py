"""
@author: Radosław Pławecki
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


class LOOCVRandomForest:
    def __init__(self, random_state=42, verbose=True):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state
        )
        self.verbose = verbose

    def run(self, X, y):
        loo = LeaveOneOut()

        y_true, y_pred, y_prob = [], [], []
        feature_importances = np.zeros(X.shape[1])

        iterator = loo.split(X)

        if self.verbose:
            iterator = tqdm(
                iterator,
                total=len(y),
                desc="LOOCV",
                leave=False
            )

        for train_idx, test_idx in iterator:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            smote = SMOTE(random_state=42, k_neighbors=1)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            self.model.fit(X_res, y_res)

            y_pred.extend(self.model.predict(X_test))
            y_prob.extend(self.model.predict_proba(X_test)[:, 1])
            y_true.extend(y_test)

            feature_importances += self.model.feature_importances_

        feature_importances /= len(y)

        return {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "y_prob": np.array(y_prob),
            "feature_importances": feature_importances
        }


class RepeatedStratifiedRF:
    def __init__(
        self,
        n_splits=5,
        n_repeats=10,
        n_estimators=100,
        random_state=42,
        use_smote=True,
        smote_k=2,
        verbose=True
    ):
        self.cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=random_state,
        )

        self.use_smote = use_smote
        self.smote_k = smote_k
        self.verbose = verbose

    def run(self, X, y):
        y_true_all, y_pred_all, y_prob_all = [], [], []

        feature_importances = np.zeros(X.shape[1])
        perm_importances = np.zeros(X.shape[1])

        total_folds = self.cv.get_n_splits()

        iterator = self.cv.split(X, y)

        if self.verbose:
            iterator = tqdm(
                iterator,
                total=total_folds,
                desc="Repeated Stratified CV",
                leave=False
            )

        for train_idx, test_idx in iterator:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.use_smote:
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=self.smote_k
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_prob)

            feature_importances += self.model.feature_importances_

            result = permutation_importance(
                self.model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42
            )
            perm_importances += result.importances_mean

        feature_importances /= total_folds
        perm_importances /= total_folds

        return {
            "y_true": np.array(y_true_all),
            "y_pred": np.array(y_pred_all),
            "y_prob": np.array(y_prob_all),
            "feature_importances": feature_importances,
            "perm_importances": perm_importances,
        }
        