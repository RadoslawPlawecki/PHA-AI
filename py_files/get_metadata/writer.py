import csv


def write_csv(path, data):
    fields = [
        "sample_id",
        "vibrant",
        "virsorter2",
        "genomad",
    ]

    data.sort(key=lambda x: int(x["sample_id"][1:]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(data)
