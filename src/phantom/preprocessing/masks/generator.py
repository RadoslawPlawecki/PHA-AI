"""
Generates CheckV-based sequence masks: TSVs listing contigs that pass
completeness/contamination thresholds, used to filter sequences downstream
during feature preprocessing.
"""

from pathlib import Path
import pandas as pd

from phantom.config.loader import ConfigLoader

QUALITY_COLS = ["contig_id", "provirus", "completeness", "contamination"]


class CheckvMaskGenerator:
    def __init__(self, config: dict):
        resolve = ConfigLoader.resolve_data_path
        self.checkv_dir = resolve(config.get("checkv", {}).get("path", "data/checkv"))
        self.masks_dir = resolve(config.get("masks", {}).get("path", "data/masks"))

    def run(self) -> int:
        """
        Generates mask TSVs for every tool subdirectory found under the
        configured CheckV root. Returns the number of tools processed.
        """
        if not self.checkv_dir.is_dir():
            print(f"[ERROR] CheckV directory not found: {self.checkv_dir}")
            return 0
        processed = 0
        for tool_dir in sorted(self.checkv_dir.iterdir()):
            if not tool_dir.is_dir():
                continue
            print(f"\n[INFO] Processing tool: {tool_dir.name}")
            merged = self._load_tool(tool_dir)
            if merged is None:
                print(f"[WARNING] No CheckV results found for {tool_dir.name}. Skipping.")
                continue
            self._write_masks(tool_dir.name, merged)
            processed += 1
        return processed

    def _load_tool(self, tool_dir: Path) -> pd.DataFrame | None:
        dfs = []
        for sample_dir in sorted(tool_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            parts = sample_dir.name.split("_")
            if len(parts) < 2:
                continue
            sample_id, tool_tag = parts[0], parts[1]
            tsv_path = sample_dir / "quality_summary.tsv"
            if not tsv_path.exists():
                continue
            df = pd.read_csv(tsv_path, sep="\t", usecols=QUALITY_COLS)
            df["contig_id"] = tool_tag + "|" + sample_id + "_" + df["contig_id"]
            dfs.append(df)
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _write_masks(self, tool: str, merged: pd.DataFrame) -> None:
        out_dir = self.masks_dir / tool
        out_dir.mkdir(parents=True, exist_ok=True)

        no_provirus = merged["provirus"] == "No"
        low_contamination = merged["contamination"] < 10
        masks = {
            "MQ_vMAGs.tsv": no_provirus & (merged["completeness"] >= 50) & low_contamination,
            "HQ_vMAGs.tsv": no_provirus & (merged["completeness"] >= 75) & low_contamination,
            "MQ_provMAGs.tsv": (merged["completeness"] >= 50) & low_contamination,
            "HQ_provMAGs.tsv": (merged["completeness"] >= 75) & low_contamination,
        }
        for filename, mask in masks.items():
            merged.loc[mask].to_csv(out_dir / filename, sep="\t", index=False)
        print(f"[INFO] Wrote {len(masks)} mask file(s) to {out_dir}")
