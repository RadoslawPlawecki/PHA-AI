"""
Preprocesses merged raw feature files per tool: applies each tool's
domain-specific cleaning/filtering before the final feature-matrix
extraction step.
"""

from pathlib import Path

from phantom.cli.features import FeatureExtractionPrompts
from phantom.config.features import FeatureConfigManager
from phantom.features.pipelines.utils import load_file, apply_mask
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


class FeaturePreprocessor:
    def __init__(self, version: str, config_mgr: FeatureConfigManager | None = None):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.raw_merged_dir = self.config_mgr.get_stage_dir(version, "raw_merged")
        self.preprocessed_dir = self.config_mgr.get_stage_dir(version, "preprocessed")
        self.prompts = FeatureExtractionPrompts()

    def run(self) -> None:
        print(f"\n[INFO] Starting feature preprocessing for version {self.version}...")
        if not self.raw_merged_dir.exists():
            print(f"[ERROR] Input directory does not exist: {self.raw_merged_dir}")
            return
        processed_any = False
        for tool, pipeline_cls in PIPELINES.items():
            tool_dir = self.raw_merged_dir / tool
            if not tool_dir.is_dir():
                continue
            for merged_file in sorted(tool_dir.glob("*.csv")):
                self._process_file(tool, pipeline_cls(), merged_file)
                processed_any = True
        if not processed_any:
            print(
                f"[ERROR] No recognized tool directories ({', '.join(PIPELINES)}) "
                f"found under {self.raw_merged_dir}"
            )

    def _process_file(self, tool: str, pipeline, merged_file: Path) -> None:
        print(f"\n[INFO] Preprocessing {tool}/{merged_file.name}...")
        df = load_file(merged_file)
        mask_path = self.prompts.ask_mask_file(merged_file)
        df = apply_mask(df, mask_path)
        preprocessed_df = pipeline.preprocess(df)
        out_dir = self.preprocessed_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{merged_file.stem}_PP.csv"
        preprocessed_df.to_csv(out_path, sep=';', index=False)
        print(f"       Saved to: {out_path} (Shape: {preprocessed_df.shape})")
