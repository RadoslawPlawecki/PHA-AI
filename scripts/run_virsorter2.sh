#!/bin/bash

shopt -s nullglob

for file in data/metagenomes/*.fa; do 
    filename=$(basename "$file") 
    
    # extract SX from 001_SX_... 
    sample=$(echo "$filename" | cut -d'_' -f2) 

    outdir="data/virsorter2/${sample}_VS2" 

    # skip if already exists 
    if [ -d "$outdir" ]; then 
        echo "Skipping $sample (already exists)" 
        continue 
    fi 
        
    echo "Processing $sample from $filename..." 

    virsorter run \
        -w "$outdir" \
        -i "$file" \
        --provirus-off --max-orf-per-seq 20 all
        
done