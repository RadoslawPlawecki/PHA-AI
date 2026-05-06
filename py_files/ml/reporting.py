"""
@author: Radosław Pławecki
"""


def format_metrics(metrics):
    return (
        "\n=== MODEL PERFORMANCE ===\n"
        f"ROC-AUC           : {metrics['roc_auc']:.3f}\n"
        f"Balanced Accuracy : {metrics['balanced_accuracy']:.3f}\n"
        f"F1 Score          : {metrics['f1']:.3f}\n"
        f"Precision         : {metrics['precision']:.3f}\n"
        f"Recall (Sensitivity): {metrics['recall']:.3f}\n"
        f"Specificity       : {metrics['specificity']:.3f}\n"
        f"G-Mean            : {metrics['geometric_mean']:.3f}\n"
        f"NPV               : {metrics['npv']:.3f}\n"
    )


def format_confusion_matrix(cm):
    return (
        "\nConfusion Matrix:\n"
        f"TP: {cm['TP']} | FP: {cm['FP']}\n"
        f"FN: {cm['FN']} | TN: {cm['TN']}\n"
    )


def format_top_features(importances, feature_names, k=5, title="Top Features"):
    lines = [f"\n=== {title} ==="]
    idx = importances.argsort()[-k:][::-1]

    for i, j in enumerate(idx, 1):
        lines.append(f"{i:>2}. {feature_names[j]:<15} | {importances[j]:.4f}")

    return "\n".join(lines)
