"""
@author: Radosław Pławecki
"""

from difflib import get_close_matches
import os
import re
import csv
               
in_root = "data/phanotate/orfs"         
out_root = "data/functions"                 

tool_tag = {
    "genomad": "geN",
    "vibrant": "VIB",
    "virsorter2": "VS2"
}


def normalize_accession(acc: str) -> str:
    return re.sub(r'k_(\d+)', r'k\1', acc.strip())


def load_metadata(prefix):
    func = f"data/modalities/1.0/preprocessed/phavip/{prefix}_ChV_PHA_ORFs_PHV_M_PP.csv"   
    comp = f"data/modalities/1.0/preprocessed/phagcn/{prefix}_ChV_PGN_M_PP.csv"              
    host = f"data/modalities/1.0/preprocessed/cherry/{prefix}_ChV_CHR_M_PP.csv"  
    annotations = {}
    genus_dict = {}
    host_dict = {}
    with open(func, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            annotations[normalize_accession(row['Accession'])] = row['Category']
    with open(comp, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            genus_dict[normalize_accession(row['Accession'])] = row['genus']
    with open(host, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            host_info = f"{row.get('ncbi_genus', '')} {row.get('ncbi_species', '')}".strip()
            if not host_info:
                host_info = "Unknown"
            host_dict[normalize_accession(row['Accession'])] = host_info
    return annotations, genus_dict, host_dict


def process_fasta():
    for tool_folder in os.listdir(in_root):
        tool_path = os.path.join(in_root, tool_folder)
        if not os.path.isdir(tool_path) or tool_folder not in tool_tag:
            continue
        prefix = tool_tag[tool_folder]
        annotations, genus_dict, host_dict = load_metadata(prefix=prefix)
        os.makedirs(out_root, exist_ok=True)
        for fasta_file in os.listdir(tool_path):
            if not fasta_file.endswith(".fasta") and not fasta_file.endswith(".fa") and not fasta_file.endswith(".fna"):
                continue
            sample_name = os.path.splitext(fasta_file)[0] + "_FUNC"
            fasta_path = os.path.join(tool_path, fasta_file)
            out_fasta_data = {}
            out_summary_data = {}
            with open(fasta_path, 'r', encoding='utf-8') as f:
                current_header = ""
                current_seq = []
                def save_current_sequence():
                    if not current_header:
                        return
                    full_accession = normalize_accession(f"{prefix}|{current_header}")
                    category = annotations.get(
                        full_accession,
                        "uncategorized"
                    )
                    contig_id = full_accession.split(':')[0]
                    g_info = genus_dict.get(contig_id, "Unknown")
                    h_info = host_dict.get(contig_id, "Unknown")
                    if category not in out_fasta_data:
                        out_fasta_data[category] = []
                        out_summary_data[category] = []
                    fasta_string = f">{current_header}\n" + "".join(current_seq) + "\n"
                    out_fasta_data[category].append(fasta_string)
                    out_summary_data[category].append([full_accession, g_info, h_info])
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        save_current_sequence() 
                        current_header = line[1:] 
                        current_seq = []
                    else:
                        current_seq.append(line)
                save_current_sequence()
            for category, seqs in out_fasta_data.items():
                cat_dir = os.path.join(out_root, tool_folder, category)
                os.makedirs(cat_dir, exist_ok=True)
                out_fasta_path = os.path.join(cat_dir, f"{sample_name}.fna")
                with open(out_fasta_path, 'w', encoding='utf-8') as out_f:
                    out_f.write("".join(seqs))
                out_tsv_path = os.path.join(cat_dir, f"{sample_name}.tsv")
                with open(out_tsv_path, 'w', encoding='utf-8', newline='') as out_t:
                    writer = csv.writer(out_t, delimiter='\t')
                    writer.writerow(["Accession", "Genus", "Host"])
                    for row_data in out_summary_data[category]:
                        writer.writerow(row_data)


if __name__ == "__main__":
    print("Sorting Data...")
    process_fasta()
    print(f"Exported to: {out_root}")

    