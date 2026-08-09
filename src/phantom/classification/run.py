"""
@author: Radosław Pławecki
"""

from ml.data.config import SingleOmicConfig, MultiOmicConfig, MdsConfig
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
            "host": "cherry",
            "func": "phavip"
        }
        tags_dict = {
            "comp": "PGN",
            "host": "CHR",
            "func": "PHV"
        }
        vtools = ["geN", "VIB", "VS2"]
        modalities = ["comp", "host", "func"]
        for vtool in vtools:
            for modality in modalities:
                config = SingleOmicConfig(
                    in_file=f"data/modalities/2.0/features/{modalities_dict[modality]}/{vtool}_{tags_dict[modality]}_FEAT.csv",
                    modality=modality,
                    vtool=vtool,
                    model_type="catboost",
                    run_loocv=True,
                    run_repeated=True,
                    use_smote=True,
                    out_dir="data/results/single_omic/2.0/catboost"
                )
                ExperimentRunner(config).run()

    @staticmethod
    def multi_omic():
        host_types = ["host=cherry", "host=iphop"]
        host_dictionary = {"cherry": "CHR", "iphop": "IPH"}
        model_types = ["catboost"]
        fusions = ["late", "early"]
        vtools = ["geN", "VIB", "VS2"]
        for host_type in host_types:
            for fusion in fusions:
                for model_type in model_types:
                    for vtool in vtools:
                        host = host_type.split("=")[1]
                        config = MultiOmicConfig(
                            comp=f"data/modalities/{host_type}/features/phagcn/{vtool}_PGN_FEAT.csv",
                            host=f"data/modalities/{host_type}/features/{host}/{vtool}_{host_dictionary[host]}_FEAT.csv",
                            func=f"data/modalities/{host_type}/features/phavip/{vtool}_PHV_FEAT.csv",
                            out_dir=f"data/results/multi_omic/{host_type}",
                            model_type=model_type,
                            run_loocv=True,
                            run_repeated=True,
                            use_smote=True,
                            fusion=fusion,
                            opt=True,
                            n_trials=250
                        )
                        ExperimentRunner(config).run()

    @staticmethod
    def mds():
        vtools = ["geN", "VIB", "VS2"]
        for vtool in vtools:
            config = MdsConfig(
                comp=f"data/modalities/features/phagcn/{vtool}_PGN_FEAT.csv",
                host=f"data/modalities/features/iphop/{vtool}_IPH_FEAT.csv",
                func=f"data/modalities/features/phavip/{vtool}_PHV_FEAT.csv",
                out_dir="data/results/mds",
            )
            ExperimentRunner(config).run()


runner = Runner()
if args.classifier == "single":
    runner.single_omic()
elif args.classifier == "multi":
    runner.multi_omic()
elif args.classifier == "mds":
    runner.mds()
