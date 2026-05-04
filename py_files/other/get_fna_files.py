"""
@author: Radosław Pławecki

The script to get the *.fna file with the assembled metagenomes for every tool (geNomad/VirSorter2/VIBRANT).
"""

from pathlib import Path
import shutil

roots = [
    "data/genomad",
    "data/vibrant",
    "data/virsorter2",
]

destinations = [
    "data/genomes/genomad",
    "data/genomes/vibrant",
    "data/genomes/virsorter2",
]

patterns = [
    "*_assembly.contigs_virus.fna",
    "*_assembly.contigs.phages_combined.fna",
    "final-viral-combined.fa",
]

for root_path, dest_path in zip(roots, destinations):
    root = Path(root_path)
    dest = Path(dest_path)
    dest.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        for file in root.rglob(pattern):
            relative_parts = file.relative_to(root).parts
            subfolder_name = relative_parts[0] if len(relative_parts) > 1 else "no_subfolder"

            new_name = f"{subfolder_name}.fna"
            print(new_name)
            dst_file = dest / new_name

            shutil.copy2(file, dst_file)
