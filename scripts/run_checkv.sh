#!/bin/bash

for file in data/genomes/vibrant/*.fna; do
    filename=$(basename "$file")      # S1_VIB.fna
    base="${filename%.fna}"           # S1_VIB
    outdir="data/checkv/vibrant/${base}_ChV"

    # skip if already exists
    if [ -d "$outdir" ]; then
        echo "Skipping $base (already exists)"
        continue
    fi

    mkdir -p "$outdir"

    echo "Processing $base..."

    checkv end_to_end \
        "$file" \
        "$outdir" \
        -d /home/raddy/PGM-AI/checkv-db-v1.5 \
        -t 16

done

for file in data/genomes/virsorter2/*.fna; do
    filename=$(basename "$file")      # S1_VS2.fna
    base="${filename%.fna}"           # S1_VS2
    outdir="data/checkv/virsorter2/${base}_ChV"

    # skip if already exists
    if [ -d "$outdir" ]; then
        echo "Skipping $base (already exists)"
        continue
    fi

    mkdir -p "$outdir"

    echo "Processing $base..."

    checkv end_to_end \
        "$file" \
        "$outdir" \
        -d /home/raddy/PGM-AI/checkv-db-v1.5 \
        -t 16

done

for file in data/genomes/genomad/*.fna; do
    filename=$(basename "$file")      # S1_geN.fna
    base="${filename%.fna}"           # S1_geN
    outdir="data/checkv/genomad/${base}_ChV"

    # skip if already exists
    if [ -d "$outdir" ]; then
        echo "Skipping $base (already exists)"
        continue
    fi

    mkdir -p "$outdir"

    echo "Processing $base..."

    checkv end_to_end \
        "$file" \
        "$outdir" \
        -d /home/raddy/PGM-AI/checkv-db-v1.5 \
        -t 16

done