import csv


def summarize_checkv(path, min_comp=50, max_cont=10):
    total = 0
    good = 0

    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")

        fields = {k.lower(): k for k in reader.fieldnames}

        comp_key = fields.get("completeness") or fields.get("estimated_completeness")
        cont_key = fields.get("contamination") or fields.get("estimated_contamination")

        if not comp_key or not cont_key:
            raise ValueError(f"Missing required columns in {path}")

        for row in reader:
            try:
                total += 1
                comp = float(row[comp_key])
                cont = float(row[cont_key])
            except (ValueError, TypeError):
                continue

            if comp >= min_comp and cont < max_cont:
                good += 1

    return total, good
