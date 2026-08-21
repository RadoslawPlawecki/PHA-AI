#!/bin/bash

for tool_path in data/phanotate/orfs/*/; do
    tool=$(basename "$tool_path")

    mkdir -p \
        data/phabox2/phavip/${tool} 

    for file in "${tool_path}"*.fna; do
        filename=$(basename "$file")
        base="${filename%.fna}"

        phb="data/phabox2/phavip/${tool}/${base}_PHV"

        echo "Processing $tool / $base..."

        if [ ! -d "$phb" ]; then
            echo "  Running PhaBOX2..."
            phabox2 \
            --task phavip \
            --dbdir phabox_db_v2_2/ \
            --outpth "$phb" \
            --contigs "$file" \
            --threads 40
        else
            echo "  Skipping PhaBOX2 (exists)"
        fi

        prediction="$phb/final_prediction/phavip_supplementary/gene_annotation.tsv"

        if [ -f "$prediction" ]; then
            mv "$prediction" "data/phabox2/phavip/${tool}/${base}_PHV.tsv"
            rm -rf "$phb"
        fi

    done
done
