#!/bin/bash

shopt -s nullglob

for file in data/megahit/*.fa; do 
    filename=$(basename "$file") 
    
    # extract SX from 001_SX_... 
    sample=$(echo "$filename" | cut -d'_' -f2) 

    outdir="data/vibrant/${sample}_VIB" 

    # skip if already exists 
    if [ -d "$outdir" ]; then 
        echo "Skipping $sample (already exists)" 
        continue 
    fi 
        
    echo "Processing $sample from $filename..." 

    python3 VIBRANT/VIBRANT_run.py \
        -i \
        "$file" \
        -t 5 \
        -folder "$outdir"
done