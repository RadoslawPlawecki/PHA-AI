#!/bin/bash

in_file="data/modalities/features/phagcn/geN_PGN_FEAT.csv"
# out_file="data/ml/sml/rf/cherry/VS2_CHR_RF.csv"

python -m project.src.pipelines.sml.main \
  --in_file "$in_file" \
  --run_fisher \
  --run_loocv \
  --run_repeated \
  --use_smote \
  --model_type catboost
