import csv


def write_csv(path, data):
    fields = [
        "sample_id",
        "vibrant",
        "virsorter2",
        "genomad",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerows(data)
