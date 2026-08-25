"""
@author: Radosław Pławecki
"""

import os
import pandas as pd
import argparse
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from phantom.classification.data.config import MdsConfig
from phantom.classification.data.data_aligner import DataAligner
from phantom.classification.data.data_loader import DataLoader
from phantom.classification.data.preprocessor import NearZeroVarianceFilter
from phantom.classification.analytics.logger import Logger
from phantom.classification.analytics.reporter import ReportFormatter
from phantom.classification.analytics.visualizer import Visualizer
from phantom.classification.analytics.saver import ExperimentSaver
from phantom.classification.ml.evaluator import EvaluatorUl


class UnsupervisedClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.saver = None
        self.X_raw = {}
        self.X_data = None
        self.labels = None
        self.common_samples = None
        self.aligned_X = {}
        self.dist_matrix_embeddings = {}
        self.mds_embeddings = {}
        self.all_metrics = {}

    def run(self):
        self._setup()
        self._load()
        self._compute_distances()
        self._evaluate()
        self._visualize()
        self._finish()

    def _setup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"run_unsupervised_{timestamp}"
        base_out_dir = getattr(self.config, 'out_dir', 'results')
        exp_dir = os.path.join(base_out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        self.logger = Logger.setup_logger(
            log_dir=exp_dir, 
            log_filename=f"{timestamp}.log"
        )
        self.saver = ExperimentSaver(exp_dir=exp_dir)
        self.logger.info("=== UNSUPERVISED LEARNING ===")
        self.saver.save_metadata(vars(self.config))

    def _load(self):
        paths = {
            "comp": self.config.comp,
            "func": self.config.func,
            "host": self.config.host,
        }
        for m in paths:
            loader = DataLoader(input_path=paths[m], logger=self.logger)
            X, labels, sample_ids = loader.load()
            nzv_filter = NearZeroVarianceFilter(logger=self.logger, threshold=4e-5)
            values, feature_names = nzv_filter.fit_transform(X)
            self.X_raw[m] = {
                "values": values,
                "feature_names": feature_names,
                "ids": list(sample_ids),
                "labels": list(labels)
            }
        self.X_data, self.labels, self.common_samples = DataAligner.align(self.X_raw)
        self.aligned_X = {m: self.X_data[m]["values"] for m in self.X_data}
        self.logger.info("Dataset Aligned")

    def _compute_distances(self):
        dist_comp = squareform(pdist(self.aligned_X["comp"], metric='jaccard'))
        dist_host = squareform(pdist(self.aligned_X["host"], metric='euclidean'))
        dist_func = squareform(pdist(self.aligned_X["func"], metric='braycurtis'))
        self.logger.info(
            f"\nDistance Matrices Computed with Dimensions:\n"
            f"Comp = {self.aligned_X['comp'].shape[1]} Features\n"
            f"Host = {self.aligned_X['host'].shape[1]} Features\n"
            f"Func = {self.aligned_X['func'].shape[1]} Features"
        )
        dist_comp_norm = dist_comp / dist_comp.max() if dist_comp.max() > 0 else dist_comp
        dist_host_norm = dist_host / dist_host.max() if dist_host.max() > 0 else dist_host
        dist_func_norm = dist_func / dist_func.max() if dist_func.max() > 0 else dist_func
        fused_distance_matrix = (dist_comp_norm + dist_host_norm + dist_func_norm) 
        self.dist_matrix_embeddings = {
            "comp": dist_comp_norm,
            "host": dist_host_norm,
            "func": dist_func_norm,
            "fused": fused_distance_matrix
        }

    def _evaluate(self):
        for name, matrix in self.dist_matrix_embeddings.items():
            self.all_metrics[name] = EvaluatorUl.evaluate(dist_matrix=matrix, labels=self.labels)
        self.logger.info(ReportFormatter.format_unsupervised_metrics(self.all_metrics))
        self.saver.save_unsupervised_metrics(self.all_metrics)

    def _visualize(self):
        self.logger.info("--- MULTI-DIMENSIONAL SCALING ---")
        mds = MDS(
            init="classical_mds", 
            n_components=2, 
            metric='precomputed', 
            random_state=42, 
            normalized_stress='auto', 
            n_init=1
        )
        for name, matrix in self.dist_matrix_embeddings.items():
            self.mds_embeddings[name] = mds.fit_transform(matrix)
        plot_path = os.path.join(self.saver.exp_dir, "mds_plot.pdf")
        Visualizer.plot_unsupervised_grid(
            coords_dict=self.mds_embeddings, 
            y_aligned=self.labels, 
            sample_ids=self.common_samples,
            save_path=plot_path
        )
        self.logger.info(f"MDS Plot Grid Saved: {plot_path}")

    def _finish(self):
        self.logger.info("=== END EXPERIMENT ===")
