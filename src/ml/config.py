"""
@author: Radosław Pławecki
"""

from dataclasses import dataclass
from pathlib import Path
import argparse

@dataclass(frozen=True)
class SingleOmicConfig:
    in_file: str
    out_file: str | None
    modality: str | None
    gtool: str | None
    run_fisher: bool
    run_loocv: bool
    run_repeated: bool
    use_smote: bool
    model_type: str
    MODALITY_MAP = {
        "host": ("iphop", "IPH"),
        "comp": ("phagcn", "PGN"),
        "func": ("phavip", "PHV"),
    }

    @classmethod
    def from_args(cls) -> "SingleOmicConfig":
        parser = argparse.ArgumentParser(description="Single-Omic Allergy Classifier")
        parser.add_argument("--in_file", type=str, default=None)
        parser.add_argument("--modality", choices=["comp", "host", "func"])
        parser.add_argument("--gtool", choices=["geN", "VIB", "VS2"])
        parser.add_argument("--out_file", type=str, default=None)
        parser.add_argument("--run_fisher", action="store_true")
        parser.add_argument("--run_loocv", action="store_true")
        parser.add_argument("--run_repeated", action="store_true")
        parser.add_argument("--use_smote", action="store_true")
        parser.add_argument("--model_type", default="rf", choices=["rf", "xgb", "catboost"])
        args = parser.parse_args()
        in_file = args.in_file
        if in_file is None:
            if args.modality is None or args.gtool is None:
                parser.error("--in_file or both --modality and --gtool are required")
            subdir, tag = cls.MODALITY_MAP[args.modality]
            in_file = str(
                Path("data/modalities/features") / subdir / f"{args.gtool}_{tag}_FEAT.csv")
        return cls(
            in_file=in_file,
            out_file=args.out_file,
            modality=args.modality,
            gtool=args.gtool,
            run_fisher=args.run_fisher,
            run_loocv=args.run_loocv,
            run_repeated=args.run_repeated,
            use_smote=args.use_smote,
            model_type=args.model_type,
        )
