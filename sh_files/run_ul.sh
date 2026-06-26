#!/bin/bash

tool="VS2"

comp="data/modalities/features/phagcn/${tool}_PGN_FEAT.csv"
host="data/modalities/features/iphop/${tool}_IPH_FEAT.csv"
func="data/modalities/features/phavip/${tool}_PHV_FEAT.csv"

python -m project.src.ml.execution.el \
  --comp "$comp" \
  --host "$host" \
  --func "$func" 
