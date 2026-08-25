"""
CLI prompts for the classification workflow.
"""

from pathlib import Path
import questionary

from phantom.config.loader import ConfigLoader
from phantom.config.features import FeatureConfigManager

RUN_TYPE_DIR_LABELS = {"single": "single-omic", "multi": "multi-omic", "unsupervised": "mds"}


class ClassificationPrompts:
    @staticmethod
    def ask_model_choice(default: str = "catboost") -> str:
        return questionary.select(
            "Select model:", choices=["catboost", "rf", "xgb"], default=default
        ).ask()

    @staticmethod
    def ask_smote_choice(default: bool = False) -> bool:
        return questionary.confirm("Apply SMOTE for class balancing?", default=default).ask()

    @staticmethod
    def ask_run_type() -> str | None:
        labels = {
            "single": "Single-Omic Classifier",
            "multi": "Multi-Omic Classifier (Fusion)",
            "unsupervised": "Unsupervised Exploration (MDS)",
        }
        choice = questionary.select("Select classification mode:", choices=list(labels.values())).ask()
        if choice is None:
            return None
        return list(labels)[list(labels.values()).index(choice)]

    @staticmethod
    def ask_validators() -> tuple[bool, bool]:
        choices = questionary.checkbox(
            "Select validation method(s) to run:",
            choices=["LOOCV", "Repeated Stratified CV"],
        ).ask() or []
        return "LOOCV" in choices, "Repeated Stratified CV" in choices

    @staticmethod
    def ask_grid_scope(run_type: str, vtools: list[str]) -> str:
        choices = ["Single File", f"All Virus Identification Tools ({'/'.join(vtools)})"]
        if run_type == "single":
            choices.append("All Virus Identification Tools x All Omics")
        return questionary.select("Select run scope:", choices=choices).ask()

    @staticmethod
    def ask_omic_choice(omics: dict) -> str:
        labels = {key: f"{val['label']} ({val['tool']})" for key, val in omics.items()}
        choice = questionary.select("Select omic:", choices=list(labels.values())).ask()
        return list(labels)[list(labels.values()).index(choice)]

    @staticmethod
    def ask_vtool_choice(vtools: list[str]) -> str:
        return questionary.select("Select Virus Identification Tool:", choices=vtools).ask()

    @staticmethod
    def ask_fusion_choice(default: str = "early") -> str:
        return questionary.select(
            "Select fusion strategy:", choices=["early", "late"], default=default
        ).ask()

    @staticmethod
    def ask_late_fusion_opt() -> tuple[bool, int]:
        opt = questionary.confirm("Optimize late-fusion weights with Optuna?", default=False).ask()
        if not opt:
            return False, 0
        n_trials = questionary.text(
            "Number of Optuna trials:", default="30",
            validate=lambda text: text.isdigit() and int(text) >= 1 or "Enter a positive integer."
        ).ask()
        return True, int(n_trials)

    @staticmethod
    def ask_classification_out_dir(run_type: str, version: str) -> Path:
        label = RUN_TYPE_DIR_LABELS[run_type]
        default_dir = ConfigLoader.resolve_data_path(f"data/results/{label}/{version}")
        answer = questionary.path(
            "Output directory for results:", default=str(default_dir), only_directories=True
        ).ask()
        return Path(answer)

    @staticmethod
    def ask_feature_version(config_mgr: FeatureConfigManager) -> str | None:
        existing_versions = config_mgr.get_existing_versions()
        if not existing_versions:
            print("[ERROR] No feature versions found in config.toml.")
            return None
        return questionary.select("Select target version:", choices=existing_versions).ask()

    @staticmethod
    def ask_omic_file(
        config_mgr: FeatureConfigManager, version: str, omic: str, omics: dict, vtool: str | None = None
    ) -> Path | None:
        tool = omics[omic]["tool"]
        tag = omics[omic]["tag"]
        extracted_dir = config_mgr.get_stage_dir(version, "extracted") / tool
        files = sorted(extracted_dir.glob(f"*_{tag}_FEAT.csv"))
        if not files:
            print(f"[ERROR] No {tag} feature files found in {extracted_dir}")
            return None
        if vtool is not None:
            match = next((f for f in files if f.name.startswith(f"{vtool}_")), None)
            if match is None:
                print(f"[ERROR] No {vtool} file for omic {omic} in {extracted_dir}")
            return match
        if len(files) == 1:
            return files[0]
        choice = questionary.select(
            f"Select {omic} ({tag}) file:", choices=[f.name for f in files]
        ).ask()
        if not choice:
            return None
        return next(f for f in files if f.name == choice)
