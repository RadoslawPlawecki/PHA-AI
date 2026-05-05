import re
from pathlib import Path
from datetime import datetime, timedelta
from .utils import extract_first_time, extract_last_time, get_virsorter2_log_time

VIBRANT_RE = re.compile(r"Runtime:\s*([0-9.]+)\s*minutes")


def runtime_vibrant(file_path: Path):
    with open(file_path) as f:
        for line in f:
            m = VIBRANT_RE.search(line)
            if m:
                return round(float(m.group(1)))
    return None


def runtime_virsorter2(folder: Path):
    start_file = folder / "log" / "iter-0" / "step1-pp" / "circular-remove-partial-gene-common.log"
    end_file = folder / "log" / "iter-0" / "step2-extract-feature" / "extract-feature-from-hmmout-common.log"

    if not start_file.exists() or not end_file.exists():
        return None
    
    start_time = get_virsorter2_log_time(start_file)
    end_time = get_virsorter2_log_time(end_file)

    if end_time < start_time:
        end_time += timedelta(days=1)

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
    