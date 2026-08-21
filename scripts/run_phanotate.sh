#!/bin/bash

for tool_path in data/genomes/*/; do
    tool=$(basename "$tool_path")

    mkdir -p \
        data/phanotate/genes/${tool} \
        data/phanotate/bed/${tool} \
        data/phanotate/orfs/${tool} \
        data/phanotate/proteins/${tool}

    for file in "${tool_path}"*.fna; do
        filename=$(basename "$file")
        base="${filename%.fna}"

        gff="data/phanotate/genes/${tool}/${base}_PHA.gff"
        bed="data/phanotate/bed/${tool}/${base}_PHA.bed"
        orf="data/phanotate/orfs/${tool}/${base}_PHA_ORFs.fna"
        prot="data/phanotate/proteins/${tool}/${base}_PHA_PROT.faa"

        echo "Processing $tool / $base..."

        # 1. Run PHANOTATE
        if [ ! -f "$gff" ]; then
            echo "  Running PHANOTATE..."
            PHANOTATE/phanotate.py "$file" > "$gff" 
        else
            echo "  Skipping PHANOTATE (exists)"
        fi

        # 2. Convert to BED
        if [ ! -f "$bed" ]; then
            echo "  Converting to BED..."
            awk 'BEGIN{OFS="\t"}
            !/^#/ {
                start=$1
                end=$2
                strand=$3
                chrom=$4

                # ensure numeric coordinates
                if (start ~ /^[0-9]+$/ && end ~ /^[0-9]+$/) {

                    if (start < end) {
                        print chrom, start-1, end, ".", ".", strand
                    } else {
                        print chrom, end-1, start, ".", ".", strand
                    }

                }
            }' "$gff" > "$bed"
        fi

        # 3. Extract ORFs
        if [ ! -f "$orf" ]; then
            echo "  Extracting ORFs..."
            bedtools getfasta \
                -fi "$file" \
                -bed "$bed" \
                -fo "$orf" \
                -s
        else
            echo "  Skipping ORF extraction (exists)"
        fi

        # 4. Translate to proteins
        if [ ! -f "$prot" ]; then
            echo "  Translating ORFs..."
            seqkit translate "$orf" > "$prot"
        fi

    done
done
