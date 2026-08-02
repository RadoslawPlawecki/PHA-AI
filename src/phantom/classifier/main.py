"""
@author: Radosław Pławecki
"""

from phantom.classifier.data.config import (
    SingleOmicConfig,
    MultiOmicConfig,
    MdsConfig
)
from phantom.classifier.execution.single_omic import SingleOmicClassifier
from phantom.classifier.execution.multi_omic import MultiOmicClassifier
from phantom.classifier.execution.unsupervised_classifier import UnsupervisedClassifier

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
