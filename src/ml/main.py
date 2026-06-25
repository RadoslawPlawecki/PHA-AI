"""
@author: Radosław Pławecki
"""

from ml.data.config import (
    SingleOmicConfig,
    MultiOmicConfig,
    MdsConfig
)
from ml.engine.single_omic import SingleOmicClassifier

class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        engine = self._build_engine()
        engine.run()

    def _build_engine(self):
        if isinstance(self.config, SingleOmicConfig):
            return SingleOmicClassifier(self.config)
        if isinstance(self.config, MultiOmicConfig):
            return MultiOmicClassifier(self.config)
        if isinstance(self.config, MdsConfig):
            return UnsupervisedClassifier(self.config)
        raise ValueError("Unknown config type")
