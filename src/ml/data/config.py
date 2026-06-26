"""
@author: Radosław Pławecki
"""

from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass(frozen=True)
class MdsConfig:
    comp: str
    host: str 
    func: str 

    @classmethod
    def from_args(cls) -> "MdSConfig":
        parser = argparse.ArgumentParser(description="Multi-Dimensional Scaling")
        parser.add_argument("--comp", type=str, required=True, help="Direct path to the virus composition modality")
        parser.add_argument("--func", type=str, required=True, help="Direct path to the functional modality")
        parser.add_argument("--host", type=str, required=True, help="Direct path to the host prediction modality")
        args = parser.parse_args()
        return cls(
            comp=args.comp,
            host=args.host,
            func=args.func,
        )


@dataclass(frozen=True)
class SingleOmicConfig:
    in_file: str
    out_dir: str
    modality: str | None
    vtool: str | None
    run_loocv: bool
    run_repeated: bool
    use_smote: bool
    model_type: str

    @classmethod
    def from_args(cls) -> "SingleOmicConfig":
        parser = argparse.ArgumentParser(description="Single-Omic Allergy Classifier")
        parser.add_argument("--in_file", type=str, default=None)
        parser.add_argument("--out_dir", type=str, required=True)
        parser.add_argument("--modality", type=str, default=None)
        parser.add_argument("--vtool", type=str, default=None)
        parser.add_argument("--run_loocv", action="store_true")
        parser.add_argument("--run_repeated", action="store_true")
        parser.add_argument("--use_smote", action="store_true")
        parser.add_argument("--model_type", default="rf", choices=["rf", "xgb", "catboost"])
        args = parser.parse_args()
        return cls(
            in_file=args.in_file,
            modality=args.modality,
            vtool=args.gtool,
            out_dir=args.out_dir,
            run_loocv=args.run_loocv,
            run_repeated=args.run_repeated,
            use_smote=args.use_smote,
            model_type=args.model_type,
        )


@dataclass(frozen=True)
class MultiOmicConfig:
    comp: str
    host: str 
    func: str 
    out_dir: str | None
    run_loocv: bool
    run_repeated: bool
    use_smote: bool
    model_type: str
    fusion: str
    opt: str | None
    n_trials: int | None

    @classmethod
    def from_args(cls) -> "MultiOmicConfig":
        parser = argparse.ArgumentParser(description="Multi-Omic Allergy Classifier")
        parser.add_argument("--comp", type=str, required=True, help="Direct path to the virus composition modality")
        parser.add_argument("--func", type=str, required=True, help="Direct path to the functional modality")
        parser.add_argument("--host", type=str, required=True, help="Direct path to the phage-host relation modality")
        parser.add_argument("--out_dir", type=str, required=True, help="Direct path to an output directory")
        parser.add_argument("--run_loocv", action="store_true", help="Run Leave-One-Out CV")
        parser.add_argument("--run_repeated", action="store_true", help="Run Repeated Stratified CV")
        parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE to training set")
        parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost"], help="Type of the model to use")
        parser.add_argument("--fusion", type=str, default="early", choices=["early", "late"], help="Type of modality fusion")
        parser.add_argument("--opt", action="store_true", help="Apply Optuma optimizer")
        parser.add_argument("--n_trials", type=int, default="30", help="Number of trials to optimize")
        args = parser.parse_args()
        return cls(
            comp=args.comp,
            host=args.host,
            func=args.func,
            out_file=args.out_file,
            run_loocv=args.run_loocv,
            run_repeated=args.run_repeated,
            use_smote=args.use_smote,
            model_type=args.model_type,
            fusion=args.fusion,
            opt=args.opt,
            n_trials=args.n_trials,
        )
