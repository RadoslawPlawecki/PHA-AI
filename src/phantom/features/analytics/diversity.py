"""
Computes per-sample taxonomic (phagcn genus) and bacterial-host (cherry
ncbi_genus) diversity from a feature version's preprocessed data, split by
the healthy/allergy patient grouping.
"""

from pathlib import Path

import pandas as pd

from phantom.config.features import FeatureConfigManager
from phantom.classification.data.labeling import Labeling
from phantom.features.pipelines.matrix import derive_patient_id
from phantom.features.analytics.visualizer import DiversityVisualizer

TOOL_FEATURE_COL = {"phagcn": "genus", "cherry": "ncbi_genus"}
TOOL_TITLES = {"phagcn": "Phage genus", "cherry": "Bacterial host genus"}
TOOL_FILE_STEMS = {"phagcn": "phagcn_genus", "cherry": "cherry_host"}
TOOL_UNIT_NAMES = {"phagcn": "genus", "cherry": "host"}


class DiversityAnalyzer:
    def __init__(self, version: str, config_mgr: FeatureConfigManager | None = None):
        self.version = version
        self.config_mgr = config_mgr or FeatureConfigManager()
        self.preprocessed_dir = self.config_mgr.get_stage_dir(version, "preprocessed")

    def _load_vtool_files(self, tool: str) -> dict[str, pd.DataFrame]:
        feature_col = TOOL_FEATURE_COL[tool]
        tool_dir = self.preprocessed_dir / tool
        loaded = {}
        if not tool_dir.is_dir():
            return loaded
        for preprocessed_file in sorted(tool_dir.glob("*.csv")):
            vtool = preprocessed_file.stem.split("_")[0]
            df = pd.read_csv(preprocessed_file, sep=';')
            df["id"] = derive_patient_id(df["Accession"])
            df["label"] = Labeling.derive_label(df["Accession"])
            loaded[vtool] = df
        return loaded

    def analyze_tool(self, tool: str) -> dict[str, dict]:
        """
        Per vtool: {"per_sample": df[id, label, genome_count, unit_count,
        ratio], "group_stats": df indexed by label (0/1) with MultiIndex
        columns (metric, "mean"/"std") for metric in [ratio, unit_count]}.

        ``unit_count`` is the number of *distinct* taxonomic units
        (occurrence-based, i.e. nunique over the feature column) present in
        a sample -- not a sum of per-unit abundance/occurrence counts.
        """
        feature_col = TOOL_FEATURE_COL[tool]
        results = {}
        for vtool, df in self._load_vtool_files(tool).items():
            per_sample = df.groupby("id").agg(
                genome_count=("Accession", "nunique"),
                unit_count=(feature_col, "nunique"),
                label=("label", "first"),
            )
            per_sample["ratio"] = per_sample["unit_count"] / per_sample["genome_count"]
            group_stats = per_sample.groupby("label")[["ratio", "unit_count"]].agg(["mean", "std"])
            results[vtool] = {
                "per_sample": per_sample.reset_index(),
                "group_stats": group_stats,
            }
        return results

    def allergy_only_units(self, tool: str) -> pd.DataFrame:
        """
        Columns: vtool, unit, n_samples, samples -- taxonomic units (phagcn
        genus / cherry host genus) whose sample-id set never intersects
        that vtool's healthy sample-id set. ``samples`` lists the actual
        sample numbers (comma-separated) the unit was found in, alongside
        the plain count in ``n_samples``.
        """
        feature_col = TOOL_FEATURE_COL[tool]
        rows = []
        for vtool, df in self._load_vtool_files(tool).items():
            healthy_ids = set(df.loc[df["label"] == 0, "id"])
            grouped = df.groupby(feature_col)["id"].agg(set)
            for unit, ids in grouped.items():
                if ids and not (ids & healthy_ids):
                    sample_numbers = sorted(int(i.split("|S")[1]) for i in ids)
                    rows.append({
                        "vtool": vtool,
                        "unit": unit,
                        "n_samples": len(sample_numbers),
                        "samples": ",".join(str(n) for n in sample_numbers),
                    })
        columns = ["vtool", "unit", "n_samples", "samples"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(["vtool", "n_samples"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def run(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        any_tool_processed = False
        for tool in ("phagcn", "cherry"):
            results = self.analyze_tool(tool)
            if not results:
                print(
                    f"[WARNING] No preprocessed {tool} data found for version "
                    f"{self.version}, skipping."
                )
                continue
            any_tool_processed = True
            group_stats_by_vtool = {vt: r["group_stats"] for vt, r in results.items()}
            unit_label = TOOL_TITLES[tool]
            file_stem = TOOL_FILE_STEMS[tool]
            DiversityVisualizer.plot_group_bar_by_vtool(
                group_stats_by_vtool, value_col="ratio",
                title=f"{unit_label} diversity per identified genome",
                ylabel=f"Avg. distinct {unit_label.lower()} / genome",
                save_path=out_dir / f"{file_stem}_per_genome.pdf",
            )
            DiversityVisualizer.plot_group_bar_by_vtool(
                group_stats_by_vtool, value_col="unit_count",
                title=f"{unit_label} diversity per sample",
                ylabel=f"Avg. distinct {unit_label.lower()} / sample",
                save_path=out_dir / f"{file_stem}_per_sample.pdf",
            )
        if not any_tool_processed:
            print(
                f"[ERROR] No preprocessed phagcn/cherry data found under "
                f"{self.preprocessed_dir}. Run preprocessing first."
            )
            return

        for tool in ("phagcn", "cherry"):
            allergy_only = self.allergy_only_units(tool)
            allergy_only_path = out_dir / f"{tool}_allergy_only_{TOOL_UNIT_NAMES[tool]}.csv"
            allergy_only.to_csv(allergy_only_path, sep=';', index=False)

        from phantom.classification.analytics.saver import ExperimentSaver
        ExperimentSaver(exp_dir=str(out_dir)).save_metadata({
            "version": self.version,
            "tools": ["phagcn", "cherry"],
        })

        print(f"\n[SUCCESS] Diversity analysis complete. Results saved to {out_dir}")
