#!/bin/bash

tool="$1"

if [[ -z "$tool" ]]; then
  echo "Usage: $0 <tool>"
  echo "Example: $0 cherry"
  echo "Example: $0 phagcn"
  echo "Example: $0 phavip"
  exit 1
fi

if [[ "$tool" == "cherry" ]]; then
  in_file="data/modalities/preprocessed/cherry/VS2_ChV_CHR_M_PP.csv"
  out_file="data/ml/sml/rf/cherry/VS2_CHR_RF.csv"

  columns=(
    Host
    ncbi_phylum
    ncbi_class
    ncbi_order
    ncbi_family
    ncbi_genus
    ncbi_species
    gtdb_phylum
    gtdb_class
    gtdb_order
    gtdb_family
    gtdb_genus
    gtdb_species
  )

elif [[ "$tool" == "phagcn" ]]; then
  in_file="data/modalities/preprocessed/phagcn/VS2_ChV_PGN_M_PP.csv"
  out_file="data/ml/sml/rf/phagcn/VS2_PGN_RF.csv"

  columns=(
    genus
    species
  )

elif [[ "$tool" == "phavip" ]]; then
  in_file="data/modalities/preprocessed/phavip/VS2_ChV_PHA_ORFs_PHV_M_PP.csv"
  out_file="data/results/sml/rf/phavip/VS2_PHV_RF.csv"

  columns=(
    Annotation
  )

else
  echo "Unknown tool: $tool"
  exit 1
fi

for column in "${columns[@]}"; do
  for min_patients in {1..4}; do

    echo "[$tool] Running with --column=$column --min_patients=$min_patients"

    python -m project.py_files.pipelines.sml.main \
      --in_file "$in_file" \
      --out_file "$out_file" \
      --column "$column" \
      --min_patients "$min_patients" \
      --run_fisher \
      --run_loocv \
      --run_repeated \
      --tool "$tool" \
      --use_smote \
      --model_type rf

  done
done