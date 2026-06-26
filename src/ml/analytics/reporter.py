"""
@author: Radosław Pławecki
"""

from collections import Counter
import pandas as pd
import numpy as np


class ReportFormatter:
    @staticmethod
    def format_metric(metric: dict) -> str:
        return f"{metric["score"]:.3f}"

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        return (
            "\nModel Performance:\n"
            f"ROC-AUC: {ReportFormatter.format_metric(metrics['roc_auc'])}\n"
            f"Balanced Accuracy: {ReportFormatter.format_metric(metrics['bacc'])}\n"
            f"F1 Score: {ReportFormatter.format_metric(metrics['f1'])}\n"
            f"Precision: {ReportFormatter.format_metric(metrics['precision'])}\n"
            f"Recall (Sensitivity): {ReportFormatter.format_metric(metrics['recall'])}\n"
            f"Specificity: {ReportFormatter.format_metric(metrics['specificity'])}\n"
            f"G-Mean: {ReportFormatter.format_metric(metrics['gmean'])}\n"
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
    def format_top_features(importances: np.ndarray, feature_names: list, X: np.ndarray, y: np.ndarray, k: int = 10) -> str:
        lines = [
            "\nTop Features & Class Characteristics:",
            f"{'No.':>3} {'Feature':<33} | {'Importance':>10} | {'Mean Class 0':>12} | {'Mean Class 1':>12}",
            "-" * 100,
        ]
        idx = importances.argsort()[-k:][::-1]
        for i, j in enumerate(idx, 1):
            feat_values = X[:, j]
            mean_0 = np.mean(feat_values[y == 0])
            mean_1 = np.mean(feat_values[y == 1])
            lines.append(
                f"{i:>3}. "
                f"{feature_names[j]:<33} | "
                f"{importances[j]:>10.4f} | "
                f"{mean_0:>12.4f} | "
                f"{mean_1:>12.4f}"
            )
        return "\n".join(lines) + "\n"

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

    @staticmethod
    def format_unsupervised_metrics(results):
        lines = [
            "",
            "Unsupervised Evaluation",
            "-" * 80,
            f"{'Modality':<10} {'Silhouette':>12} {'ANOSIM R':>12} {'ANOSIM p':>12} {'PERMANOVA F':>14} {'PERMANOVA p':>14}",
            "-" * 80,
        ]
        for modality, metrics in results.items():
            lines.append(
                f"{modality.upper():<10}"
                f"{metrics['silhouette']['score']:>12.3f}"
                f"{metrics['anosim']['statistic']:>12.3f}"
                f"{metrics['anosim']['p_value']:>12.3f}"
                f"{metrics['permanova']['statistic']:>14.3f}"
                f"{metrics['permanova']['p_value']:>14.3f}"
            )
        lines.append("-" * 80)
        return "\n".join(lines)
