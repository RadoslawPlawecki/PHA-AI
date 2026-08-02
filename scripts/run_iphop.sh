#!/bin/bash

set -euo pipefail

CLEANUP=false

source ~/miniconda3/etc/profile.d/conda.sh

# create environment only if it does not exist
if ! conda env list | grep -q "^iphop_env "; then
    echo "Creating conda environment..."

    conda create -y -n iphop_env python=3.8
    conda install -y -c conda-forge -c bioconda iphop

else
    echo "Activating environment..."
fi

conda activate iphop_env

if [ ! -d "iphop_db/Jun_2025_pub_rw" ]; then
    echo "Downloading database..."

    mkdir -p iphop_db

    iphop download \
        --db_dir iphop_db/ \
        -dbv iPHoP_db_Jun25_rw

    iphop download \
        --db_dir iphop_db/Jun_2025_pub_rw \
        --full_verify

else
    echo "Database already exists"
fi

for tool_path in data/post-checkv/*/; do

    tool=$(basename "$tool_path")

    mkdir -p "data/iphop/${tool}"

    for file in "${tool_path}"/*.fna; do

        filename=$(basename "$file")
        base="${filename%.fna}"

        iph="data/iphop/${tool}/${base}_IPH"

        echo "Processing $tool / $base..."

        # Skip if output already exists
        if [ ! -d "$iph" ]; then

            echo "  Running iPHoP..."

            iphop predict \
                --fa_file "$file" \
                --db_dir iphop_db/Jun_2025_pub_rw \
                --out_dir "$iph"

        else
            echo "  Skipping iPHoP (exists)"
        fi

    done
done

if [ "$CLEANUP" = true ]; then

    echo "Cleaning up..."

    conda deactivate || true

    conda env remove -y -n iphop_env

    rm -rf iphop_db

    echo "Cleanup complete"
fi