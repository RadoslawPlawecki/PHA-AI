import re
from pathlib import Path
from datetime import datetime

TIME_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")


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


def get_sample_id(folder: Path):
    return folder.name.split("_")[0]


def get_virsorter2_log_time(log_file):
    with open(log_file) as f:
        first_line = f.readline()

    start_str = first_line.split("]")[0].strip("[")
    start_str = " ".join(start_str.split()[:2])
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M")
