#!/bin/bash

set -euo pipefail
shopt -s nullglob

# Map tool → merged file
declare -A tool_files=(
    ["genomad"]="geN_ChV_PHA_PROT.faa"
    ["vibrant"]="VIB_ChV_PHA_PROT.faa"
    ["virsorter2"]="VS2_ChV_PHA_PROT.faa"
)

for tool in "${!tool_files[@]}"; do
    echo "=== Running vContact2 for $tool ==="

    prot="data/phanotate/merged/${tool_files[$tool]}"
    map_dir="data/vcontact2/proteins-fp/$tool"
    out_dir="data/vcontact2/output/$tool"

    mkdir -p "$map_dir" "$out_dir"

    filename=$(basename "$prot")
    base="${filename%.faa}"

    clean_prot="$map_dir/${base}_clean.faa"
    map="$map_dir/${base}_proteins.csv"
    vout="$out_dir/${base}_vC2"

    echo "Processing $tool / $base..."

    # -------------------------
    # 1. Prepare FASTA + mapping
    # -------------------------
    if [ ! -f "$clean_prot" ] || [ "$prot" -nt "$clean_prot" ]; then
        echo "  Preparing vContact2 input..."

        > "$clean_prot"
        echo "protein_id,contig_id" > "$map"

        awk -v map="$map" -v clean="$clean_prot" '
        /^>/ {
            contig=$0
            sub(/^>/, "", contig)
            sub(/:.*/, "", contig)

            counts[contig]++
            protein_id = contig "_" counts[contig]

            print ">" protein_id >> clean
            print protein_id "," contig >> map
            next
        }
        {
            print >> clean
        }
        ' "$prot"

    else
        echo "  Skipping prep (exists)"
    fi

    # -------------------------
    # 2. Run vContact2
    # -------------------------
    if [ ! -d "$vout" ]; then
        echo "  Running vContact2..."

        vcontact2 \
            --raw-proteins "$clean_prot" \
            --proteins-fp "$map" \
            --db 'ProkaryoticViralRefSeq94-Merged' \
            --rel-mode Diamond \
            --pcs-mode MCL \
            --vcs-mode ClusterONE \
            --c1-bin ~/bin/cluster_one-1.0.jar \
            --output-dir "$vout"

    else
        echo "  Skipping vContact2 (exists)"
    fi

done