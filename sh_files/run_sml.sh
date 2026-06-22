#!/bin/bash

out_file="data/ml/sml/SML_CB.csv"

modalities=(comp host func)
gtools=(geN VIB VS2 ALL)

for gtool in "${gtools[@]}"; do
  for modality in "${modalities[@]}"; do
        echo "Running modality=${modality}, gtool=${gtool}"
        python -m project.src.ml.sml \
            --modality "$modality" \
            --gtool "$gtool" \
            --out_file "$out_file" \
            --run_fisher \
            --run_loocv \
            --run_repeated \
            --use_smote \
            --model_type catboost
    done
done
