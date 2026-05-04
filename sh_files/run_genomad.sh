#!/bin/bash

for file in data/metagenomes/*.fa; do
    filename=$(basename "$file")

    # extract SX from 001_SX_...
    sample=$(echo "$filename" | cut -d'_' -f2)

    outdir="data/genomad/${sample}_geN"

    # skip if already exists
    if [ -d "$outdir" ]; then
        echo "Skipping $sample (already exists)"
        continue
    fi

    echo "Processing $sample from $filename..."

    genomad end-to-end --cleanup --splits 8 \
        "$file" \
        "$outdir" \
        genomad_db

done