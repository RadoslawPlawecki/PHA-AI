"""
@author: Radosław Pławecki
"""

from ml.data.config import SingleOmicConfig
from ml.main import ExperimentRunner


def main():
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
                run_fisher=True,
                run_loocv=True,
                run_repeated=True,
                use_smote=True,
                out_dir="data/results"
            )
            ExperimentRunner(config).run()


if __name__ == "__main__":
    main()