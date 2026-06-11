#!/bin/bash

in_file="data/modalities/features/phavip/VS2_PHV_FEAT.csv"
# out_file="data/ml/sml/rf/cherry/VS2_CHR_RF.csv"

python -m project.src.ml.sml \
  --in_file "$in_file" \
  --run_fisher \
  --run_loocv \
  --run_repeated \
  --use_smote \
  --model_type catboost
