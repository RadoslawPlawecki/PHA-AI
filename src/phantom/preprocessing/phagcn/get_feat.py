"""
@author: Radosław Pławecki
"""

from pathlib import Path
from .feature_extractor import PhagcnFeatureExtractor
from phantom.cli.prompts import FeatureExtractionPrompts, ModalityFileSelector
from phantom.preprocessing.utils import load_file
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_feature_extraction():
    try:
        selector = ModalityFileSelector(modality="phagcn")
        in_file = selector.select()
        if not in_file:
            logger.warning("No input file selected. Exiting.")
            return
        preprocessed_dir = selector.get_stage_dir("preprocessed")
        features_dir = selector.get_stage_dir("features")
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        prefix = in_file.stem[:3]
        preprocessed_out = preprocessed_dir / f"{prefix}_ChV_PGN_M_PP.csv"
        features_out = features_dir / f"{prefix}_PGN_FEAT.csv"
        prompts = FeatureExtractionPrompts()
        mask_path = prompts.ask_mask_file(in_file)
        min_patients = prompts.ask_min_patients()
        binary = prompts.ask_binary()
        extractor = PhagcnFeatureExtractor(
            min_phagcn_score=0.5, 
            min_patients=min_patients, 
            binary=binary
        )
        logger.info("Loading and preprocessing data...")
        df = load_file(in_file)
        df_preprocessed = extractor.preprocess(df)
        feature_col = prompts.ask_column(df_preprocessed)
        logger.info("Extracting features...")    
        feature_matrix = extractor.process_file(
            in_file=in_file, 
            preprocessed_out_path=preprocessed_out,
            features_out_path=features_out,
            feature_col=feature_col,
            mask_path=mask_path
        )
        print(feature_matrix.head())
        logger.info(f"Extraction completed! Features saved to: {features_out}")
    except Exception as e:
        logger.error(f"An error occurred during feature extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_feature_extraction()
