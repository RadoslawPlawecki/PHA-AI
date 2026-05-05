import re
from pathlib import Path
from datetime import datetime

TIME_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")


def build_index(tool_root):
    root = Path(tool_root)
    index = {}
    for folder in root.iterdir():
        if folder.is_dir():
            sid = get_sample_id(folder)
            index[sid] = folder
    return index


def build_megahit_index(root="data/megahit"):
    root = Path(root)
    index = {}
    for file in root.glob("*_assembly.contigs.fa"):
        sid = get_sample_id(file)
        index[sid] = file
    return index


def build_checkv_index(root):
    root = Path(root)
    index = {}
    for folder in root.iterdir():
        if folder.is_dir():
            sid = get_sample_id(folder)
            tsv = folder / "quality_summary.tsv"
            if tsv.exists():
                index[sid] = tsv
    return index


def get_sample_id(path):
    name = Path(path).name
    m = re.search(r"S\d+", name)
    if m:
        return m.group(0)
    raise ValueError(f"Cannot extract sample_id from {name}")


def get_file_size(path):
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)  
    return size_bytes, round(size_mb, 2)


def extract_first_time(file_path):
    with open(file_path) as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                return m.group(1)
    return None


def extract_last_time(file_path):
    last = None
    with open(file_path) as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                last = m.group(1)
    return last


def get_virsorter2_log_time(log_file):
    with open(log_file) as f:
        first_line = f.readline()

    start_str = first_line.split("]")[0].strip("[")
    start_str = " ".join(start_str.split()[:2])
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M")
