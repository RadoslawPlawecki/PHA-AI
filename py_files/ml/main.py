"""
@author: Radosław Pławecki
"""

from .logger import setup_logger
from .data_loader import DataLoader
from .analyze_fisher import FisherAnalyzer
from .random_forest import LOOCVRandomForest, RepeatedStratifiedRF
from .reporter import (
    format_metrics,
    format_confusion_matrix,
    format_top_features
)
from .evaluator import Evaluator
from .csv_reporter import CSVReporter
import os
from tqdm import tqdm


def main():
    input_path = "data/ml/"
    roots_dict = {
        "genomad": "geN",
        "virsorter2": "VS2",
        "vibrant": "VIB"
    }


    def run_model(X, y, feature_names, model, feature_importance, logger):
        results = model.run(X, y)

        metrics = Evaluator.evaluate(
            results["y_true"],
            results["y_pred"],
            results["y_prob"]
        )

        logger.info(format_metrics(metrics))
        logger.info(format_confusion_matrix(metrics["confusion_matrix"]))
        logger.info(
            format_top_features(
                feature_importance(results),
                feature_names
            )
        )

        return metrics, results


    for i in range(2, 11):
        for root in roots_dict.keys():
            tag = roots_dict[root]

            logger = setup_logger(f"{tag}_mP{i}")

            filepath = os.path.join(input_path, root, f"{tag}_vC2_mP{i}.csv")

            logger.info("=== START EXPERIMENT ===")

            logger.info("Loading data...")
            loader = DataLoader(filepath)
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

            loocv_model = LOOCVRandomForest(random_state=42, 
                                            n_estimators=300, 
                                            max_depth=3, 
                                            min_samples_leaf=2, 
                                            use_smote=True, 
                                            smote_k=1, 
                                            verbose=True)

            metrics_loocv, results_loocv = run_model(
                X,
                y,
                feature_names,
                loocv_model,
                lambda r: r["feature_importances"],
                logger
            )

            # --- REPEATED CV ---
            logger.info("Running Repeated Stratified CV...")

            repeated_model = RepeatedStratifiedRF(n_splits=2, 
                                                  n_repeats=2, 
                                                  n_estimators=300, 
                                                  max_depth=3, 
                                                  min_samples_leaf=2, 
                                                  random_state=42,
                                                  use_smote=True,
                                                  smote_k=2,
                                                  verbose=True
                                                  )

            metrics_repeated, results_repeated = run_model(
                X,
                y,
                feature_names,
                repeated_model,
                lambda r: r["perm_importances"],
                logger
            )

            logger.info("=== END EXPERIMENT ===")

            CSVReporter.save_metrics(
                    filepath=f"results/random_forest/RF_SMOTE_mP{i}.csv",
                    experiment_name=f"{root}_loocv",
                    metrics=metrics_loocv
                )

            CSVReporter.save_metrics(
                    filepath=f"results/random_forest/RF_SMOTE_mP{i}.csv",
                    experiment_name=f"{root}_repeated",
                    metrics=metrics_repeated
                )


if __name__ == "__main__":
    main()