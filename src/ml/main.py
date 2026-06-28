"""
@author: Radosław Pławecki
"""

from ml.data.config import (
    SingleOmicConfig,
    MultiOmicConfig,
    MdsConfig
)
from ml.execution.single_omic import SingleOmicClassifier
from ml.execution.multi_omic import MultiOmicClassifier
from ml.execution.unsupervised_classifier import UnsupervisedClassifier

class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        classifier = self._build_classifier()
        classifier.run()

    def _build_classifier(self):
        if isinstance(self.config, SingleOmicConfig):
            return SingleOmicClassifier(self.config)
        if isinstance(self.config, MultiOmicConfig):
            return MultiOmicClassifier(self.config)
        if isinstance(self.config, MdsConfig):
            return UnsupervisedClassifier(self.config)
        raise ValueError("Unknown config type")
