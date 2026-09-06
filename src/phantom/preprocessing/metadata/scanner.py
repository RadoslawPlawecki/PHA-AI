"""
Scans output files from geNomad, VIBRANT, and VirSorter2 to extract runtime 
information.
"""

import re
import csv
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from phantom.config.loader import ConfigLoader

TIME_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
VIBRANT_RE = re.compile(r"Runtime:\s*([0-9.]+)\s*minutes")


def get_sample_id(path):
    name = Path(path).name
    m = re.search(r"S\d+", name)
    if m: return m.group(0)
    raise ValueError(f"Cannot extract sample_id from {name}")


def get_file_size(path):
    size_bytes = path.stat().st_size
    return size_bytes, round(size_bytes / (1024 * 1024), 2)


def extract_first_time(file_path):
    with open(file_path) as f:
        for line in f:
            m = TIME_RE.search(line)
            if m: return m.group(1)
    return None


def extract_last_time(file_path):
    last = None
    with open(file_path) as f:
        for line in f:
            m = TIME_RE.search(line)
            if m: last = m.group(1)
    return last


def get_virsorter2_log_time(log_file):
    with open(log_file) as f:
        first_line = f.readline()
    start_str = first_line.split("]")[0].strip("[")
    start_str = " ".join(start_str.split()[:2])
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M")


def summarize_checkv(path, medium_comp=50, high_comp=75, max_cont=10):
    total = medium = high = 0
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = {k.lower(): k for k in reader.fieldnames}
        comp_key = fields.get("completeness") or fields.get("estimated_completeness")
        cont_key = fields.get("contamination") or fields.get("estimated_contamination")
        if not comp_key or not cont_key:
            return None, None, None
        for row in reader:
            try:
                comp = float(row[comp_key])
                cont = float(row[cont_key])
            except (ValueError, TypeError):
                continue
            total += 1
            if cont >= max_cont:
                continue
            if comp >= medium_comp:
                medium += 1
            if comp >= high_comp:
                high += 1
    return total, medium, high


def build_index(root):
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir(): 
        print(f"[WARNING] Tool directory not found: '{root_path}'. Skipping "
              "runtimes for this path.")
        return {}
    return {get_sample_id(f): f for f in root_path.iterdir() if f.is_dir()}

def build_megahit_index(root):
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir(): 
        print(f"[WARNING] MEGAHIT directory not found: '{root_path}'. "
              "Skipping size metrics.")
        return {}
    return {get_sample_id(f): f for f in root_path.glob("*_assembly.contigs.fa")}

def build_checkv_index(root):
    root_path = Path(root)
    index = {}
    if not root_path.exists() or not root_path.is_dir(): 
        print(f"[WARNING] CheckV directory not found: '{root_path}'. "
              "Skipping quality control metrics for this tool.")
        return index
    for folder in root_path.iterdir():
        if folder.is_dir():
            tsv = folder / "quality_summary.tsv"
            if tsv.exists():
                index[get_sample_id(folder)] = tsv
    return index


def runtime_vibrant(file_path: Path):
    with open(file_path) as f:
        for line in f:
            m = VIBRANT_RE.search(line)
            if m: return round(float(m.group(1)))
    return None


def runtime_virsorter2(folder: Path):
    start_file = folder / "log" / "iter-0" / "step1-pp" / "circular-remove-partial-gene-common.log"
    end_file = folder / "log" / "iter-0" / "step2-extract-feature" / "extract-feature-from-hmmout-common.log"
    if not start_file.exists() or not end_file.exists(): return None
    start_time = get_virsorter2_log_time(start_file)
    end_time = get_virsorter2_log_time(end_file)
    if end_time < start_time: end_time += timedelta(days=1)
    return round((end_time - start_time).total_seconds() / 60)


def runtime_genomad(folder):
    start_files = list(folder.glob("*contigs_annotate.log"))
    end_files = list(folder.glob("*contigs_summary.log"))
    if not start_files or not end_files: return None
    start = datetime.strptime(extract_first_time(start_files[0]), "%H:%M:%S")
    end = datetime.strptime(extract_last_time(end_files[0]), "%H:%M:%S")
    if end < start: end += timedelta(days=1)
    return round((end - start).total_seconds() / 60)


def scan_metadata(config: dict) -> pd.DataFrame:
    """
    Scans predefined roots for metadata and formats it as a DataFrame 
    joined on id.
    
    NOTE: Log parsing for runtimes are strictly hardcoded for the utilized 
    versions of geNomad, VIBRANT and VirSorter2. Custom tools added to the 
    config are ignored during this step due to the specific nature of various 
    outputs.
    """
    resolve = ConfigLoader.resolve_data_path
    vib_path = resolve(config.get("vibrant", {}).get("path", "data/vibrant"))
    vs2_path = resolve(config.get("virsorter2", {}).get("path", "data/virsorter2"))
    gen_path = resolve(config.get("genomad", {}).get("path", "data/genomad"))
    megahit_path = resolve(config.get("megahit", {}).get("path", "data/megahit"))
    checkv_base = resolve(config.get("checkv", {}).get("path", "data/checkv"))

    vib_root = build_index(vib_path)
    vs2_root = build_index(vs2_path)
    gen_root = build_index(gen_path)
    megahit_root = build_megahit_index(megahit_path)
    
    checkv_vib = build_checkv_index(checkv_base / "vibrant")
    checkv_vs2 = build_checkv_index(checkv_base / "virsorter2")
    checkv_gen = build_checkv_index(checkv_base / "genomad")

    sample_ids = set(vib_root) | set(vs2_root) | set(gen_root) | set(megahit_root)
    results = {sid: {"id": sid} for sid in sample_ids}

    for sid, folder in vib_root.items():
        logs = list(folder.rglob("VIBRANT_log_run_*_assembly.contigs.log"))
        if logs: results[sid]["vib_runtime"] = runtime_vibrant(max(logs, key=lambda f: f.stat().st_mtime))
    for sid, folder in vs2_root.items():
        results[sid]["vs2_runtime"] = runtime_virsorter2(folder)
    for sid, folder in gen_root.items():
        results[sid]["gen_runtime"] = runtime_genomad(folder)

    for sid, file in megahit_root.items():
        b, mb = get_file_size(file)
        results[sid]["megahit_size_bytes"] = b
        results[sid]["megahit_size_mb"] = mb

    for tag, index in (("vib", checkv_vib), ("vs2", checkv_vs2), ("gen", checkv_gen)):
        for sid, path in index.items():
            t, m, h = summarize_checkv(path)
            if t is not None:
                results[sid][f"checkv_{tag}_total"] = t
                results[sid][f"checkv_{tag}_medium"] = m
                results[sid][f"checkv_{tag}_high"] = h

    return pd.DataFrame(list(results.values()))
