#!/bin/bash

comp="data/modalities/features/phagcn/VIB_PGN_FEAT.csv"
host="data/modalities/features/iphop/VIB_IPH_FEAT.csv"
func="data/modalities/features/phavip/VIB_PHV_FEAT.csv"
# out_file="data/ml/sml/rf/cherry/VS2_CHR_RF.csv"

python -m project.src.ml.late_fusion \
  --comp "$comp" \
  --host "$host" \
  --func "$func" \
  --run_loocv \
  --run_repeated \
  --use_smote \
  --model_type catboost
