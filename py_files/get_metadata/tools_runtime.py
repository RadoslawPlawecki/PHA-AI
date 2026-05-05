import re
from pathlib import Path
from datetime import datetime, timedelta
from .utils import extract_first_time, extract_last_time

VIBRANT_RE = re.compile(r"Runtime:\s*([0-9.]+)\s*minutes")


def runtime_vibrant(file_path: Path):
    with open(file_path) as f:
        for line in f:
            m = VIBRANT_RE.search(line)
            if m:
                return round(float(m.group(1)))
    return None


def runtime_virsorter2(folder: Path):
    log_file = folder / "log" / "iter-0" / "step1-pp" / "circular-remove-partial-gene-common.log"

    if not log_file.exists():
        return None

    with open(log_file) as f:
        first_line = f.readline()

    start_str = first_line.split("]")[0].strip("[")
    start_str = " ".join(start_str.split()[:2])

    start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
    end_time = datetime.fromtimestamp(folder.stat().st_mtime)

    return round((end_time - start_time).total_seconds() / 60)


def runtime_genomad(folder):
    start_files = list(folder.glob("*contigs_annotate.log"))
    end_files = list(folder.glob("*contigs_summary.log"))

    if not start_files or not end_files:
        return None

    start_file = start_files[0]
    end_file = end_files[0]

    start = datetime.strptime(extract_first_time(start_file), "%H:%M:%S")
    end = datetime.strptime(extract_last_time(end_file), "%H:%M:%S")

    if end < start:
        end += timedelta(days=1)

    return round((end - start).total_seconds() / 60)
    