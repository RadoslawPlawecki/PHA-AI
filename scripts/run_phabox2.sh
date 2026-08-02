#!/bin/bash

for tool_path in data/post-checkv/*/; do
    tool=$(basename "$tool_path")

    mkdir -p \
        data/phabox2/phagcn/${tool} 

    for file in "${tool_path}"*.fna; do
        filename=$(basename "$file")
        base="${filename%.fna}"

        phb="data/phabox2/phagcn/${tool}/${base}_PHB"

        echo "Processing $tool / $base..."

        # 1. Run PHANOTATE
        if [ ! -d "$phb" ]; then
            echo "  Running PhaBOX2..."
            phabox2 \
            --task phagcn \
            --dbdir phabox_db_v2_2/ \
            --outpth "$phb" \
            --contigs "$file" \
            --threads 40
        else
            echo "  Skipping PhaBOX2 (exists)"
        fi

    done
done
