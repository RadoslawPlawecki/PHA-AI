"""
Controls the classification workflow: single-omic, multi-omic (fusion),
and unsupervised (MDS) exploration.
"""

from pathlib import Path

from phantom.config.loader import ConfigLoader
from phantom.config.features import FeatureConfigManager
from phantom.cli.classification import ClassificationPrompts
from phantom.classification.data.config import SingleOmicConfig, MultiOmicConfig, MdsConfig
from phantom.classification.execution.single_omic import SingleOmicClassifier
from phantom.classification.execution.multi_omic import MultiOmicClassifier
from phantom.classification.execution.unsupervised_classifier import UnsupervisedClassifier


class ClassificationController:
    def __init__(self, config_path: Path | str | None = None):
        loader = ConfigLoader(Path(config_path) if config_path else None)
        self.config = loader.load()
        self.config_path = loader.path
        self.config_mgr = FeatureConfigManager(self.config_path)
        self.omics = self.config["omics"]
        self.vtools = list(self.config["tools"].keys())

    def run(self):
        run_type = ClassificationPrompts.ask_run_type()
        if not run_type:
            return
        if run_type == "single":
            self._run_single_omic()
        elif run_type == "multi":
            self._run_multi_omic()
        elif run_type == "unsupervised":
            self._run_unsupervised()

    def _vtools_for_scope(self, scope: str) -> list[str | None]:
        if scope == "Single file":
            return [ClassificationPrompts.ask_vtool_choice(self.vtools)]
        return list(self.vtools)

    def _run_single_omic(self):
        version = ClassificationPrompts.ask_feature_version(self.config_mgr)
        if not version:
            return
        scope = ClassificationPrompts.ask_grid_scope("single", self.vtools)
        model_type = ClassificationPrompts.ask_model_choice()
        run_loocv, run_repeated = ClassificationPrompts.ask_validators()
        use_smote = ClassificationPrompts.ask_smote_choice()
        out_dir = ClassificationPrompts.ask_classification_out_dir("single", version)

        runs: list[tuple[Path, str, str | None]] = []
        if scope == "All virus-identification tools x all omics":
            for omic in self.omics:
                for vtool in self.vtools:
                    in_file = ClassificationPrompts.ask_omic_file(self.config_mgr, version, omic, self.omics, vtool=vtool)
                    if in_file:
                        runs.append((in_file, omic, vtool))
        else:
            omic = ClassificationPrompts.ask_omic_choice(self.omics)
            for vtool in self._vtools_for_scope(scope):
                in_file = ClassificationPrompts.ask_omic_file(self.config_mgr, version, omic, self.omics, vtool=vtool)
                if in_file:
                    runs.append((in_file, omic, vtool))

        for in_file, omic, vtool in runs:
            config = SingleOmicConfig(
                in_file=str(in_file),
                out_dir=str(out_dir),
                modality=omic,
                vtool=vtool,
                run_loocv=run_loocv,
                run_repeated=run_repeated,
                use_smote=use_smote,
                model_type=model_type,
            )
            SingleOmicClassifier(config).run()

    def _run_multi_omic(self):
        version = ClassificationPrompts.ask_feature_version(self.config_mgr)
        if not version:
            return
        scope = ClassificationPrompts.ask_grid_scope("multi", self.vtools)
        model_type = ClassificationPrompts.ask_model_choice()
        run_loocv, run_repeated = ClassificationPrompts.ask_validators()
        use_smote = ClassificationPrompts.ask_smote_choice()
        fusion = ClassificationPrompts.ask_fusion_choice()
        opt, n_trials = (False, 0)
        if fusion == "late":
            opt, n_trials = ClassificationPrompts.ask_late_fusion_opt()
        out_dir = ClassificationPrompts.ask_classification_out_dir("multi", version)

        for vtool in self._vtools_for_scope(scope):
            comp = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "comp", self.omics, vtool=vtool)
            host = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "host", self.omics, vtool=vtool)
            func = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "func", self.omics, vtool=vtool)
            if not (comp and host and func):
                continue
            config = MultiOmicConfig(
                comp=str(comp), host=str(host), func=str(func),
                out_dir=str(out_dir), run_loocv=run_loocv, run_repeated=run_repeated,
                use_smote=use_smote, model_type=model_type, fusion=fusion,
                opt=opt, n_trials=n_trials,
            )
            MultiOmicClassifier(config).run()

    def _run_unsupervised(self):
        version = ClassificationPrompts.ask_feature_version(self.config_mgr)
        if not version:
            return
        scope = ClassificationPrompts.ask_grid_scope("unsupervised", self.vtools)
        out_dir = ClassificationPrompts.ask_classification_out_dir("unsupervised", version)

        for vtool in self._vtools_for_scope(scope):
            comp = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "comp", self.omics, vtool=vtool)
            host = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "host", self.omics, vtool=vtool)
            func = ClassificationPrompts.ask_omic_file(self.config_mgr, version, "func", self.omics, vtool=vtool)
            if not (comp and host and func):
                continue
            config = MdsConfig(comp=str(comp), host=str(host), func=str(func), out_dir=str(out_dir))
            UnsupervisedClassifier(config).run()


if __name__ == "__main__":
    ClassificationController().run()
