"""
@author: Radosław Pławecki
"""

from ml.data.config import SingleOmicConfig, MultiOmicConfig
from ml.main import ExperimentRunner
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--classifier", choices=["mds", "single", "multi"])
args = parser.parse_args()


class Runner():
    @staticmethod
    def single_omic():
        modalities_dict = {
            "comp": "phagcn",
            "host": "iphop",
            "func": "phavip"
        }
        tags_dict = {
            "comp": "PGN",
            "host": "IPH",
            "func": "PHV"
        }
        vtools = ["geN", "VIB", "VS2"]
        modalities = ["comp", "host", "func"]
        for vtool in vtools:
            for modality in modalities:
                config = SingleOmicConfig(
                    in_file=f"data/modalities/features/{modalities_dict[modality]}/{vtool}_{tags_dict[modality]}_FEAT.csv",
                    modality=modality,
                    vtool=vtool,
                    model_type="catboost",
                    run_loocv=True,
                    run_repeated=True,
                    use_smote=True,
                    out_dir="data/results"
                )
                ExperimentRunner(config).run()

    @staticmethod
    def multi_omic():
        comp="data/modalities/features/phagcn/geN_PGN_FEAT.csv"
        host="data/modalities/features/iphop/geN_IPH_FEAT.csv"
        func="data/modalities/features/phavip/geN_PHV_FEAT.csv"
        vtools = ["geN", "VIB", "VS2"]
        for vtool in vtools:
            config = MultiOmicConfig(
                comp=f"data/modalities/features/phagcn/{vtool}_PGN_FEAT.csv",
                host=f"data/modalities/features/iphop/{vtool}_IPH_FEAT.csv",
                func=f"data/modalities/features/phavip/{vtool}_PHV_FEAT.csv",
                out_dir="data/results/multi_omic",
                model_type="catboost",
                run_loocv=True,
                run_repeated=True,
                use_smote=True,
                fusion="late",
                opt=True,
                n_trials=100
            )
            ExperimentRunner(config).run()


runner = Runner()
if args.classifier == "single":
    runner.single_omic()
elif args.classifier == "multi":
    runner.multi_omic()
