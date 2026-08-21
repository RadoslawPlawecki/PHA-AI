"""
Builds final feature matrices from preprocessed per-tool feature data.
"""

from pathlib import Path
import pandas as pd

from phantom.cli.features import FeatureExtractionPrompts
from phantom.config.features import FeatureConfigManager
from phantom.features.pipelines.tools.cherry import CherryFeaturePipeline
from phantom.features.pipelines.tools.phagcn import PhagcnFeaturePipeline
from phantom.features.pipelines.tools.phavip import PhavipFeaturePipeline
from phantom.features.pipelines.tools.phatyp import PhatypFeaturePipeline

PIPELINES = {
    "cherry": CherryFeaturePipeline,
    "phagcn": PhagcnFeaturePipeline,
    "phavip": PhavipFeaturePipeline,
    "phatyp": PhatypFeaturePipeline,
}


class FeatureExtractor:
    def __init__(self, version: str, config_mgr: FeatureConfigManager | None = None):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.preprocessed_dir = self.config_mgr.get_stage_dir(version, "preprocessed")
        self.extracted_dir = self.config_mgr.get_stage_dir(version, "extracted")
        self.prompts = FeatureExtractionPrompts()

    def run(self) -> None:
        print(f"\n[INFO] Starting feature extraction for version {self.version}...")
        if not self.preprocessed_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.preprocessed_dir}")
            return
        processed_any = False
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.preprocessed_dir / tool
            if not tool_dir.is_dir():
                continue
            for preprocessed_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), preprocessed_file)
                processed_any = True
        if not processed_any:
            print(
                f"[ERROR] No preprocessed data found under {self.preprocessed_dir}. "
                f"Run preprocessing first."
            )

    def _process_file(self, tool: str, pipeline, preprocessed_file: Path) -> None:
        print(f"\n[INFO] Extracting features for {tool}/{preprocessed_file.name}...")
        df = pd.read_csv(preprocessed_file, sep=';')
        feature_col = self.prompts.ask_column(df) if pipeline.NEEDS_FEATURE_COLUMN else None
        feature_matrix = pipeline.build_feature_matrix(df, feature_col)
        out_dir = self.extracted_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{preprocessed_file.stem}_FEAT.csv"
        feature_matrix.to_csv(out_path, sep=';', index=False)
        print(f"       Saved to: {out_path} (Shape: {feature_matrix.shape})")
