#!/bin/bash

set -euo pipefail
shopt -s nullglob

# -------------------------
# Tool → merged FAA file
# -------------------------
declare -A tool_files=(
    ["genomad"]="geN_ChV_PHA_PROT.faa"
    ["vibrant"]="VIB_ChV_PHA_PROT.faa"
    ["virsorter2"]="VS2_ChV_PHA_PROT.faa"
)

# -------------------------
# Tool → prefix
# -------------------------
declare -A tool_prefix=(
    ["genomad"]="geN"
    ["vibrant"]="VIB"
    ["virsorter2"]="VS2"
)

ALL_TOOLS=("genomad" "vibrant" "virsorter2")

# -------------------------
# Run per-tool + merged
# -------------------------
for tool in "${ALL_TOOLS[@]}" "all"; do

    echo "=== Running vContact2 for $tool ==="

    map_dir="data/vcontact2/proteins-fp/$tool"
    out_dir="data/vcontact2/output/$tool"

    mkdir -p "$map_dir" "$out_dir"

    # -------------------------
    # Define base name
    # -------------------------
    if [[ "$tool" == "all" ]]; then
        base="ChV_PHA_PROT"
    else
        filename=$(basename "data/phanotate/merged/${tool_files[$tool]}")
        base="${filename%.faa}"
    fi

    clean_prot="$map_dir/${base}_clean.faa"
    map="$map_dir/${base}_proteins.csv"
    vout="$out_dir/${base}_vC2"

    echo "Processing $tool / $base..."

    # =========================================================
    # 1. Build FASTA + mapping
    # =========================================================

    if [[ "$tool" == "all" ]]; then

        rebuild=false

        if [[ ! -f "$clean_prot" ]]; then
            rebuild=true
        else
            for t in "${ALL_TOOLS[@]}"; do
                file="data/phanotate/merged/${tool_files[$t]}"
                if [[ "$file" -nt "$clean_prot" ]]; then
                    rebuild=true
                    break
                fi
            done
        fi

        if [[ "$rebuild" == true ]]; then

            echo "  Merging + preparing FASTA..."

            > "$clean_prot"
            echo "protein_id,contig_id" > "$map"

            for t in "${ALL_TOOLS[@]}"; do

                file="data/phanotate/merged/${tool_files[$t]}"
                prefix="${tool_prefix[$t]}"

                awk -v prefix="$prefix" \
                    -v map="$map" \
                    -v clean="$clean_prot" '
                /^>/ {

                    contig=$0

                    # remove >
                    sub(/^>/, "", contig)

                    # remove protein suffix after :
                    sub(/:.*/, "", contig)

                    # normalize contig names
                    # S10_k_149_2 → S10_k149_2
                    gsub("_k_", "_k", contig)

                    # prepend tool prefix
                    contig = prefix "|" contig

                    # unique protein IDs
                    counts[contig]++
                    protein_id = contig "_" counts[contig]

                    print ">" protein_id >> clean
                    print protein_id "," contig >> map

                    next
                }

                {
                    print >> clean
                }
                ' "$file"

            done

        else
            echo "  Skipping merge/prep (up-to-date)"
        fi

    else

        prot="data/phanotate/merged/${tool_files[$tool]}"

        if [[ ! -f "$clean_prot" ]] || [[ "$prot" -nt "$clean_prot" ]]; then

            echo "  Preparing FASTA..."

            > "$clean_prot"
            echo "protein_id,contig_id" > "$map"

            awk -v prefix="${tool_prefix[$tool]}" \
                -v map="$map" \
                -v clean="$clean_prot" '
            /^>/ {

                contig=$0

                # remove >
                sub(/^>/, "", contig)

                # remove protein suffix after :
                sub(/:.*/, "", contig)

                # normalize contig names
                # S10_k_149_2 → S10_k149_2
                gsub("_k_", "_k", contig)

                # prepend tool prefix
                contig = prefix "|" contig

                # unique protein IDs
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
            echo "  Skipping prep (up-to-date)"
        fi
    fi

    # =========================================================
    # 2. Run vContact2
    # =========================================================

    if [[ ! -d "$vout" ]]; then

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

    echo
done

echo "All vContact2 runs completed."