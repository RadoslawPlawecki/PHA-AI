"""
@author: Radosław Pławecki
"""

import pandas as pd
import os


checkv_path = "./data/checkv/"
output_base = "./data/post-checkv/"

tool_name_map = {
    "genomad": "geN_ChV.fna",
    "virsorter2": "VS2_ChV.fna",
    "vibrant": "VIB_ChV.fna"
}

os.makedirs(output_base, exist_ok=True)


def parse_fasta(fasta_path):
    """Simple FASTA parser -> dict {header: sequence}"""
    records = {}
    header = None
    seq_lines = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records[header] = "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header:
            records[header] = "".join(seq_lines)

    return records


def write_fasta(records, out_path):
    with open(out_path, "w") as f:
        for h, seq in records.items():
            f.write(f">{h}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


for tool in os.listdir(checkv_path):
    tool_path = os.path.join(checkv_path, tool)

    if not os.path.isdir(tool_path):
        continue

    print(f"\nProcessing tool: {tool}")

    out_dir = os.path.join(output_base, tool)
    os.makedirs(out_dir, exist_ok=True)

    merged_records = {}

    for sample in os.listdir(tool_path):
        sample_id = sample.split("_")[0]  # keeps only 'S1'
        sample_path = os.path.join(tool_path, sample)

        if not os.path.isdir(sample_path):
            continue

        tsv_path = os.path.join(sample_path, "quality_summary.tsv")
        fasta_path = os.path.join(sample_path, "viruses.fna")

        if not os.path.exists(tsv_path) or not os.path.exists(fasta_path):
            continue

        df = pd.read_csv(
            tsv_path,
            sep="\t",
            usecols=["contig_id", "provirus", "completeness", "contamination"]
        )

        df_filtered = df[
            (df["provirus"] == "No") &
            (df["completeness"] >= 50) &
            (df["contamination"] < 10)
        ]

        keep_ids = set(df_filtered["contig_id"])

        print(f"{tool}/{sample}: kept {len(keep_ids)} contigs")

        fasta_records = parse_fasta(fasta_path)

        filtered_records = {}

        for h, seq in fasta_records.items():
            if h in keep_ids:
                # modify header: k149_123 → k_149_123
                if h.startswith("k"):
                    h_mod = "k_" + h[1:]
                else:
                    h_mod = h

                new_header = f"{sample_id}_{h_mod}"
                filtered_records[new_header] = seq

        # write per-sample output
        out_fasta = os.path.join(out_dir, f"{sample}.fna")
        write_fasta(filtered_records, out_fasta)

        # add to merged
        merged_records.update(filtered_records)

    # write merged file per tool
    out_name = tool_name_map.get(tool, f"{tool}_merged.fna")

    merged_out = os.path.join(output_base, out_name)
    # write_fasta(merged_records, merged_out)

    print(f"Merged file written: {merged_out}")
