"""
@author: Radosław Pławecki
"""


def format_metric(metric):

    score = metric["score"]

    if "ci" in metric:
        ci = metric["ci"]

        return (
            f"{score:.3f} "
            f"(95% CI: {ci['lower']:.3f}–{ci['upper']:.3f})"
        )

    return f"{score:.3f}"


def format_metrics(metrics):
    return (
        "\n=== MODEL PERFORMANCE ===\n"
        f"ROC-AUC: {format_metric(metrics['roc_auc'])}\n"
        f"Balanced Accuracy: {format_metric(metrics['balanced_accuracy'])}\n"
        f"F1 Score: {format_metric(metrics['f1'])}\n"
        f"Precision: {format_metric(metrics['precision'])}\n"
        f"Recall (Sensitivity): {format_metric(metrics['recall'])}\n"
        f"Specificity: {format_metric(metrics['specificity'])}\n"
        f"G-Mean: {format_metric(metrics['geometric_mean'])}\n"
        f"NPV: {format_metric(metrics['npv'])}\n"
        f"Average Precision Score: {format_metric(metrics['pr_auc'])}\n"
        f"Matthews Correlation Coefficient: {format_metric(metrics['mcc'])}\n"
    )


def format_confusion_matrix(cm):
    return (
        "\nConfusion Matrix:\n"
        f"TP: {cm['TP']} | FP: {cm['FP']}\n"
        f"FN: {cm['FN']} | TN: {cm['TN']}\n"
    )


def format_top_features(importances, feature_names, k=10, title="Top Features"):
    lines = [f"\n=== {title} ==="]
    idx = importances.argsort()[-k:][::-1]

    for i, j in enumerate(idx, 1):
        lines.append(f"{i:>2}. {feature_names[j]:<15} | {importances[j]:.4f}")

    return "\n".join(lines)
