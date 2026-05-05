from pathlib import Path
from .config import ROOTS
from .tools_runtime import runtime_vibrant, runtime_virsorter2, runtime_genomad
from .utils import get_sample_id


def build_index(tool_root):
    root = Path(tool_root)
    index = {}

    for folder in root.iterdir():
        if folder.is_dir():
            sid = get_sample_id(folder)
            index[sid] = folder

    return index


def scan():
    vibrant_root = build_index("data/vibrant")
    vs2_root = build_index("data/virsorter2")
    gen_root = build_index("data/genomad")

    sample_ids = set(vibrant_root) | set(vs2_root) | set(gen_root)

    results = {
        sid: {
            "sample_id": sid,
            "vibrant": None,
            "virsorter2": None,
            "genomad": None,
        }
        for sid in sample_ids
    }

    for sid, folder in vibrant_root.items():
        run_logs = list(folder.rglob("VIBRANT_log_run_*_assembly.contigs.log"))
        if run_logs:
            results[sid]["vibrant"] = runtime_vibrant(max(run_logs, key=lambda f: f.stat().st_mtime))

    for sid, folder in vs2_root.items():
        results[sid]["virsorter2"] = runtime_virsorter2(folder)

    for sid, folder in gen_root.items():
        results[sid]["genomad"] = runtime_genomad(folder)

    return list(results.values())
