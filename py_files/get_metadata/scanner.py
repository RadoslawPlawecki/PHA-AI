from pathlib import Path
from .config import ROOTS
from .tools_runtime import runtime_vibrant, runtime_virsorter2, runtime_genomad
from .utils import *
from .checkv_counter import summarize_checkv
 

def scan():
    vibrant_root = build_index("data/vibrant")
    vs2_root = build_index("data/virsorter2")
    gen_root = build_index("data/genomad")
    megahit_root = build_megahit_index("data/megahit")
    checkv_vibrant = build_checkv_index("data/checkv/vibrant")
    checkv_vs2 = build_checkv_index("data/checkv/virsorter2")
    checkv_gen = build_checkv_index("data/checkv/genomad")

    sample_ids = set(vibrant_root) | set(vs2_root) | set(gen_root) | set(megahit_root)

    results = {
        sid: {
            "sample_id": sid,
            "megahit_size_bytes": None,
            "megahit_size_mb": None,
            "vib_runtime": None,
            "vs2_runtime": None,
            "gen_runtime": None,
            "checkv_vib_total": None,
            "checkv_vib_good": None,
            "checkv_vs2_total": None,
            "checkv_vs2_good": None,
            "checkv_gen_total": None,
            "checkv_gen_good": None,
        }
        for sid in sample_ids
    }

    for sid, folder in vibrant_root.items():
        run_logs = list(folder.rglob("VIBRANT_log_run_*_assembly.contigs.log"))
        if run_logs:
            results[sid]["vib_runtime"] = runtime_vibrant(max(run_logs, key=lambda f: f.stat().st_mtime))

    for sid, folder in vs2_root.items():
        results[sid]["vs2_runtime"] = runtime_virsorter2(folder)

    for sid, folder in gen_root.items():
        results[sid]["gen_runtime"] = runtime_genomad(folder)

    for sid, file in megahit_root.items():
        size_bytes, size_mb = get_file_size(file)
        results[sid]["megahit_size_bytes"] = size_bytes
        results[sid]["megahit_size_mb"] = size_mb

    for sid, path in checkv_vibrant.items():
        total, good = summarize_checkv(path)
        results[sid]["checkv_vib_total"] = total
        results[sid]["checkv_vib_good"] = good

    for sid, path in checkv_vs2.items():
        total, good = summarize_checkv(path)
        results[sid]["checkv_vs2_total"] = total
        results[sid]["checkv_vs2_good"] = good

    for sid, path in checkv_gen.items():
        total, good = summarize_checkv(path)
        results[sid]["checkv_gen_total"] = total
        results[sid]["checkv_gen_good"] = good

    return list(results.values())
