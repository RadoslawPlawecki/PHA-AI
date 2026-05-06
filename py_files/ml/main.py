"""
@author: Radosław Pławecki
"""

from .logger import setup_logger
from .data_loader import DataLoader
from .analyze_fisher import FisherAnalyzer
from .random_forest import LOOCVRandomForest, RepeatedStratifiedRF
from .reporting import (
    format_metrics,
    format_confusion_matrix,
    format_top_features
)
from .evaluator import Evaluator
from tqdm import tqdm


logger = setup_logger("VirSorter2")


def main():
    logger.info("=== START EXPERIMENT ===")

    logger.info("Loading data...")
    loader = DataLoader('data/ml/VS2_vC2.csv')
    X, y, feature_names = loader.load()

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # --- FISHER ---
    logger.info("Running Fisher's Exact Test...")
    fisher = FisherAnalyzer()
    significant = fisher.run(X, y, feature_names)

    if not significant:
        logger.info("No significant features found (after FDR).")
    else:
        logger.info("Significant features:")
        for name, p in significant:
            logger.info(f"{name} | p={p:.4e}")

    # --- LOOCV ---
    logger.info("Running LOOCV Random Forest...")
    model = LOOCVRandomForest(verbose=True)

    results = model.run(X, y)

    logger.info("Evaluating LOOCV...")
    metrics = Evaluator.evaluate(
        results["y_true"],
        results["y_pred"],
        results["y_prob"]
    )

    logger.info(format_metrics(metrics))
    logger.info(format_confusion_matrix(metrics["confusion_matrix"]))
    logger.info(format_top_features(results["feature_importances"], feature_names))

    # --- REPEATED CV ---
    logger.info("Running Repeated Stratified CV...")
    model = RepeatedStratifiedRF(verbose=True)

    results = model.run(X, y)

    logger.info("Evaluating Repeated CV...")
    metrics = Evaluator.evaluate(
        results["y_true"],
        results["y_pred"],
        results["y_prob"]
    )

    logger.info(format_metrics(metrics))
    logger.info(format_confusion_matrix(metrics["confusion_matrix"]))
    logger.info(format_top_features(
        results["perm_importances"],
        feature_names,
        title="Top Features (Permutation)"
    ))

    logger.info("=== END EXPERIMENT ===")


if __name__ == "__main__":
    main()