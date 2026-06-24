"""
@author: Radosław Pławecki
"""

import csv
import os


class CSVReporter:
    @staticmethod
    def save_metrics(filepath, experiment_name, metrics):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        metric_mapping = {
            "roc_auc": "auc",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "specificity": "specificity",
            "geometric_mean": "gmean",
            "npv": "nvp",
            "pr_auc": "pr_auc",
            "mcc": "mcc",
        }
        file_exists = os.path.isfile(filepath)
        rows = []
        for metric_key, metric_name in metric_mapping.items():
            rows.append({
                "metric": metric_name,
                experiment_name: metrics[metric_key]["score"]
            })
        # if file does not exist -> create new
        if not file_exists:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["metric", experiment_name]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return
        # read existing file
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        # existing columns
        fieldnames = reader[0].keys()
        # add new experiment column if needed
        if experiment_name not in fieldnames:
            fieldnames = list(fieldnames) + [experiment_name]
        # update rows
        metric_to_score = {
            metric_mapping[k]: metrics[k]["score"]
            for k in metric_mapping
        }
        for row in reader:
            metric_name = row["metric"]
            if metric_name in metric_to_score:
                row[experiment_name] = metric_to_score[metric_name]
        # save updated CSV
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
