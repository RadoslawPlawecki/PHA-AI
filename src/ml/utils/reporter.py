"""
@author: Radosław Pławecki
"""

from collections import Counter
import pandas as pd
import numpy as np


class ReportFormatter:
    @staticmethod
    def format_metric(metric: dict) -> str:
        score = metric["score"]
        if "ci" in metric:
            ci = metric["ci"]
            return f"{score:.3f} (95% CI: {ci['lower']:.3f}–{ci['upper']:.3f})"
        return f"{score:.3f}"

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        return (
            "\nModel Performance:\n"
            f"ROC-AUC: {ReportFormatter.format_metric(metrics['roc_auc'])}\n"
            f"Balanced Accuracy: {ReportFormatter.format_metric(metrics['balanced_accuracy'])}\n"
            f"F1 Score: {ReportFormatter.format_metric(metrics['f1'])}\n"
            f"Precision: {ReportFormatter.format_metric(metrics['precision'])}\n"
            f"Recall (Sensitivity): {ReportFormatter.format_metric(metrics['recall'])}\n"
            f"Specificity: {ReportFormatter.format_metric(metrics['specificity'])}\n"
            f"G-Mean: {ReportFormatter.format_metric(metrics['geometric_mean'])}\n"
            f"NPV: {ReportFormatter.format_metric(metrics['npv'])}\n"
            f"Average Precision Score: {ReportFormatter.format_metric(metrics['pr_auc'])}\n"
            f"Matthews Correlation Coefficient: {ReportFormatter.format_metric(metrics['mcc'])}\n"
        )

    @staticmethod
    def format_confusion_matrix(cm: dict) -> str:
        return (
            "\nConfusion Matrix:\n"
            f"TP: {cm['TP']:02d} | FP: {cm['FP']:02d}\n"
            f"FN: {cm['FN']:02d} | TN: {cm['TN']:02d}\n"
        )

    @staticmethod
    def format_top_features(importances: np.ndarray, feature_names: list, k: int = 10) -> str:
        lines = ["\nTop Features:"]
        idx = importances.argsort()[-k:][::-1]
        for i, j in enumerate(idx, 1):
            lines.append(f"{i:>2}. {feature_names[j]:<15} | {importances[j]:.4f}")
        return "\n".join(lines)

    @staticmethod
    def format_misclassified_samples(y_true: np.ndarray, y_pred: np.ndarray, 
                                     test_idx: np.ndarray, sample_ids: pd.Series, top_k: int = 10) -> str:
        fp_indices = test_idx[(y_true == 0) & (y_pred == 1)]
        fn_indices = test_idx[(y_true == 1) & (y_pred == 0)]
        fp_counts = Counter(fp_indices)
        fn_counts = Counter(fn_indices)
        def format_block(title: str, counts: Counter) -> str:
            if not counts:
                return f"{title}: None"
            parts = []
            for idx, count in counts.most_common():
                label = str(sample_ids.iloc[idx])
                if count > 1:
                    parts.append(f"{label} [{count}]")
                else:
                    parts.append(label)
            return f"{title}: " + ", ".join(parts)
        fp_line = format_block("False Positives", fp_counts)
        fn_line = format_block("False Negatives", fn_counts)
        return "\nMisclassified Samples:\n" + fp_line + "\n" + fn_line + "\n"
