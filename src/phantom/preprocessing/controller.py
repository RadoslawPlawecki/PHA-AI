"""
Controls the preprocessing workflow: collect genomes, gather metadata.
"""

import sys
import pandas as pd
from pathlib import Path

from phantom.config.loader import ConfigLoader
from phantom.cli.preprocessing import PreprocessingPrompts
from phantom.preprocessing.metadata.mapping import MetadataManager
from phantom.preprocessing.collection.genomes import GenomeCollector
from phantom.preprocessing.metadata.scanner import scan_metadata
from phantom.preprocessing.metadata.labeling import (
    ManualLabeler, PatternLabeler, SampleThresholdLabeler, ExternalCSVLabeler
)

class PreprocessingController:
    def __init__(self):
        self.config = ConfigLoader().load()
        meta_path = self.config.get("genomes", {}).get(
            "metadata_path", "data/genomes/genome_metadata.csv"
        )
        self.meta_mapper = MetadataManager(ConfigLoader.resolve_data_path(meta_path))

    def run(self):
        while True:
            print("\n--- Preprocessing ---")
            action = PreprocessingPrompts.ask_action()
            if action is None or action == "Back":
                break
            if action.startswith("1)"):
                df_collected = self._collect_genomes()
                if not df_collected.empty:
                    print("\n[INFO] Collection complete. Triggering metadata "
                          "gathering...")
                    self._gather_metadata(df_collected)
            elif action.startswith("2)"):
                self._gather_metadata()

    def _collect_genomes(self) -> pd.DataFrame:
        collector = GenomeCollector(self.config, self.meta_mapper)
        print("\n[INFO] Scanning directories for putative viral genomes...")
        invalid_count = collector.scan_for_invalid()
        auto_format = False
        if invalid_count > 0:
            auto_format = PreprocessingPrompts.ask_auto_format(invalid_count)
            if auto_format is None: 
                return pd.DataFrame()
        print("[INFO] Copying files into the central registry...\n")
        df = collector.collect(auto_format=auto_format)
        if df.empty:
            print("[ERROR] No genomes were found or copied.")
        return df

    def _label_genomes(self, df: pd.DataFrame) -> pd.DataFrame:
        choice = PreprocessingPrompts.ask_strategy(len(df))
        if not choice: 
            return pd.DataFrame()
        strategy = None
        if choice.startswith("1)"):
            selected = PreprocessingPrompts.ask_interactive(df["id"].tolist())
            if selected is not None: 
                strategy = ManualLabeler(selected)
        elif choice.startswith("2)"):
            threshold = PreprocessingPrompts.ask_threshold()
            if threshold is not None: 
                strategy = SampleThresholdLabeler(threshold)
        elif choice.startswith("3)"):
            pattern = PreprocessingPrompts.ask_pattern()
            if pattern: 
                strategy = PatternLabeler(pattern)
        elif choice.startswith("4)"):
            csv_path = PreprocessingPrompts.ask_external_csv()
            if csv_path: 
                strategy = ExternalCSVLabeler(Path(csv_path))
        if not strategy:
            print("\n[ERROR] Labeling was aborted.")
            return pd.DataFrame()
        labels_map = strategy.assign_labels(df)
        df_labels = pd.DataFrame(list(labels_map.items()), columns=["id", "label"])
        positives = df_labels["label"].sum() if not df_labels.empty else 0
        print(f"\n[SUCCESS] Labeling Complete! Processed {len(labels_map)} "
              f"genomes ({positives} positives).")
        return df_labels

    def _gather_metadata(self, pre_collected_df: pd.DataFrame = None):
        targets = PreprocessingPrompts.ask_metadata_targets()
        if not targets: return
        dfs_to_merge = []
        if any(t.startswith("1)") for t in targets):
            if pre_collected_df is not None and not pre_collected_df.empty:
                dfs_to_merge.append(pre_collected_df)
            else:
                print("[ERROR] Gathering renaming data requires collection "
                      "first.")
                collected = self._collect_genomes()
                if not collected.empty: 
                    dfs_to_merge.append(collected)
        if any(t.startswith("2)") for t in targets):
            base_df = pre_collected_df if (pre_collected_df is not None and not pre_collected_df.empty) else self.meta_mapper.load()
            if base_df.empty:
                print("[ERROR] Cannot label: No samples found in registry.")
            else:
                labels = self._label_genomes(base_df)
                if not labels.empty: 
                    dfs_to_merge.append(labels)
        needs_scan = (any(t.startswith(prefix) for prefix in ("3)", "4)", "5)") for t in targets))
        if needs_scan:
            print("\n[INFO] Scanning for logs, sizes, and CheckV results (geNomad, VIBRANT, VirSorter2 only)...")
            df_scan = scan_metadata(self.config)    
            if not df_scan.empty:
                scan_mappings = {
                    "3)": ["vib_runtime", "vs2_runtime", "gen_runtime"],
                    "4)": ["checkv_vib_total", "checkv_vib_good", 
                           "checkv_vs2_total", "checkv_vs2_good", 
                           "checkv_gen_total", "checkv_gen_good"],
                    "5)": ["megahit_size_bytes", "megahit_size_mb"]
                }
                for prefix, cols in scan_mappings.items():
                    if any(t.startswith(prefix) for t in targets):
                        valid_cols = ["id"] + [c for c in cols if c in df_scan.columns]
                        if len(valid_cols) > 1:  
                            dfs_to_merge.append(df_scan[valid_cols])
        for df_part in dfs_to_merge:
            if not df_part.empty and "id" in df_part.columns:
                try:
                    self.meta_mapper.add_metadata_columns(df_part)
                except ValueError as e:
                    print(f"[ERROR] {e}")
        if dfs_to_merge:
            print("\n[SUCCESS] Metadata gathering and update complete!")


if __name__ == "__main__":
    PreprocessingController().run()
