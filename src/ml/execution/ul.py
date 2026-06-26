"""
@author: Radosław Pławecki
"""

from ml.ingestion.config import MdsConfig
from ml.ingestion.data_aligner import DataAligner
from ml.ingestion.data_loader import DataLoader
from ml.ingestion.preprocessor import NearZeroVarianceFilter
from ml.analytics.logger import Logger
from ml.analytics.reporter import ReportFormatter
from ml.analytics.csv_reporter import CSVReporter
from ml.analytics.visualizer import plot_unsupervised_grid
from ml.ml.evaluator import EvaluatorUl
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


def main():
    config = MdsConfig.from_args()
    logger = Logger.setup_logger('vs2_mds')
    logger.info("=== UNSUPERVISED LEARNING ===")
    paths = {
        "comp": config.comp,
        "func": config.func,
        "host": config.host,
    }
    X_raw = {}
    for m in paths:
        loader = DataLoader(input_path=paths[m], logger=logger)
        X, labels, sample_ids = loader.load()
        nzv_filter = NearZeroVarianceFilter(logger=logger, threshold=4e-5)
        values, feature_names = nzv_filter.fit_transform(X)
        X_raw[m] = {
            "values": values,
            "feature_names": feature_names,
            "ids": list(sample_ids),
            "labels": list(labels)
        }
    X_data, y_aligned, common_samples = DataAligner.align(X_raw)
    logger.info(f"Dataset aligned")
    aligned_X = {m: X_data[m]["values"] for m in X_data}
    dist_comp = squareform(pdist(aligned_X["comp"], metric='jaccard'))
    dist_host = squareform(pdist(aligned_X["host"], metric='euclidean'))
    dist_func = squareform(pdist(aligned_X["func"], metric='braycurtis'))
    logger.info(f"\nDistance matrices computed with dimensions:\n"
                f"Comp = {aligned_X['comp'].shape[1]} features\n"
                f"Host = {aligned_X['host'].shape[1]} features\n"
                f"Func = {aligned_X['func'].shape[1]} features")
    dist_comp_norm = dist_comp / dist_comp.max() if dist_comp.max() > 0 else dist_comp
    dist_host_norm = dist_host / dist_host.max() if dist_host.max() > 0 else dist_host
    dist_func_norm = dist_func / dist_func.max() if dist_func.max() > 0 else dist_func
    fused_distance_matrix = (dist_comp_norm + dist_host_norm + dist_func_norm) 
    dist_matrix_embeddings = {
        "comp": dist_comp_norm,
        "host": dist_host_norm,
        "func": dist_func_norm,
        "fused": fused_distance_matrix
    }
    all_metrics = {}
    for name, matrix  in dist_matrix_embeddings.items():
        all_metrics[name] = EvaluatorUl.evaluate(dist_matrix=matrix, labels=y_aligned)
    logger.info(ReportFormatter.format_unsupervised_metrics(all_metrics))
    logger.info("--- MULTI-DIMENSIONAL SCALING ---")
    mds = MDS(init="classical_mds", n_components=2, metric='precomputed', random_state=42, normalized_stress='auto', n_init=1)
    coords_comp = mds.fit_transform(dist_comp_norm)
    coords_host = mds.fit_transform(dist_host_norm)
    coords_func = mds.fit_transform(dist_func_norm)
    coords_fused = mds.fit_transform(fused_distance_matrix)
    mds_embeddings = {
        "comp": coords_comp,
        "host": coords_host,
        "func": coords_func,
        "fused": coords_fused
    }
    plot_unsupervised_grid(coords_dict=mds_embeddings, y_aligned=y_aligned, sample_ids=common_samples)
    logger.info("MDS plot grid generated")
    logger.info("=== END EXPERIMENT ===")
    

if __name__ == "__main__":
   main()
