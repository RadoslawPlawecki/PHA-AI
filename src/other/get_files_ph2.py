"""
@author: Radosław Pławecki
The script to move the specific files for different PhaBOX2 tools.
"""

from pathlib import Path
import shutil

gtools = ["genomad", "virsorter2", "vibrant"]
ptools = ["phavip", "cherry", "phagcn", "phatyp"]

patterns = {
    "phavip": "gene_annotation.tsv",
    "cherry": "cherry_prediction.tsv",
    "phagcn": "phagcn_prediction.tsv",
    "phatyp": "phatyp_prediction.tsv"
}

suffixes = {
    "phavip": "PHV",
    "cherry": "CHR",
    "phagcn": "PGN",
    "phatyp": "PHT"
}

jobs = [
    {
        "root": f"data/phabox2/{ptool}/{gtool}",
        "dest": f"data/modalities/raw/{ptool}/{gtool}",
        "pattern": patterns[ptool],
        "suffix": suffixes[ptool]
    }
    for ptool in ptools
    for gtool in gtools
]


for job in jobs:
    root = Path(job["root"])
    dest = Path(job["dest"])
    pattern = job["pattern"]
    dest.mkdir(parents=True, exist_ok=True)
    for file in root.rglob(pattern):
        if "final_prediction" not in file.parts:
            continue
        rel = file.relative_to(root)
        if len(rel.parts) > 1:
            subfolder_name = rel.parts[0]
        else:
            subfolder_name = file.parent.name
        parts = subfolder_name.split("_")
        parts[-1] = job["suffix"]
        name = "_".join(parts)
        new_name = f"{name}{file.suffix}"
        dst_file = dest / new_name
        print(f"{file} -> {dst_file}")
        shutil.copy2(file, dst_file)
