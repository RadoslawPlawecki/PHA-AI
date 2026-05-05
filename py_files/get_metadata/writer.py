import csv


def write_csv(path, data):
    fields = [
        "sample_id",
        "megahit_size_bytes",
        "megahit_size_mb",
        "vib_runtime",
        "vs2_runtime",
        "gen_runtime",
        "checkv_vib_total",
        "checkv_vib_good",
        "checkv_vs2_total",
        "checkv_vs2_good",
        "checkv_gen_total",
        "checkv_gen_good",
    ]

    data.sort(key=lambda x: int(x["sample_id"][1:]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(data)
