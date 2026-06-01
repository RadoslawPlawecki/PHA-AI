"""
@author: Radosław Pławecki
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional

class PhavipFeatureExtractor:
    CATEGORY_PATTERNS = {
        "structural":
        r"capsid|head|tail|portal|fiber|baseplate|"
        r"sheath|spike|scaffolding|neck|virion|coat|"
        r"tape measure|connector|pilot",

        "packaging":
            r"terminase|packaging",

        "lysis":
            r"holin|endolysin|amidase|lysin|spanin|"
            r"peptidoglycan|murein",

        "lysogeny":
            r"integrase|repressor|anti-repressor|cro|"
            r"cII|rdf|directionality factor|excisionase|invertase",

        "replication_recombination":
            r"polymerase|primase|helicase|ligase|"
            r"ssdna|dna binding|annealing|recT|recE|"
            r"gam-like|sak4|erf|resolvase|recombination|"
            r"endonuclease|exonuclease|holliday|swi2/snf2|"
            r"replisome|dna repair",

        "transcription_regulation":
            r"transcription|rna polymerase|sigma|"
            r"transcriptional activator|transcriptional regulator|"
            r"helix-turn-helix|hth|gemA",

        "host_takeover":
            r"histone-like|hu/ihf|rnapi associated|"
            r"nuclear pore|chromatin|host takeover",

        "nucleotide_metabolism":
            r"dutp|diphosphatase|ribonucleotide reductase|"
            r"kinase|phosphatase|queuosine|sam|"
            r"cyclohydrolase",

        "defense_accessory":
            r"methyltransferase|restriction|toxin|antitoxin|"
            r"reverse transcriptase|parA|parB|transposase",

        "maturation_accessory":
            r"protease|peptidase",

        "metabolism_accessory":
            r"isomerase|hydrolase|oxidoreductase|reductase",

        "unknown":
            r"duf|hypothetical|orf|gp\d+|vog"
    }

    def __init__(self, min_coverage=0.7, min_pident=0.35):
        self.min_coverage = min_coverage
        self.min_pident = min_pident
        self.compiled_patterns = {
            k: re.compile(v, re.IGNORECASE)
            for k, v in self.CATEGORY_PATTERNS.items()
        }

    def load_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='warn')
        gtool_id = path.stem.split('_')[0]
        df["Accession"] = self.format_accession(gtool_id, df["Genome"])
        df["ORF"] = self.format_accession(gtool_id, df["ORF"])
        df['id'] = df['Accession'].astype(str).str.split('_').str[0]
        return df

    @staticmethod
    def format_accession(prefix: str, column: pd.Series) -> pd.Series:
        return prefix + "|" + column.str.replace(r'k_', 'k', regex=False)

    @staticmethod
    def extract_labels(samples: pd.Series) -> pd.Series:
        last_part = samples.str.split("|").str[-1]
        numeric = (
            last_part
            .str.extract(r'S(\d+)')[0]
            .fillna(999)
            .astype(int)
        )
        return (numeric <= 34).astype(int)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df["pident"] = pd.to_numeric(df["pident"], errors="coerce")
        df["label"] = self.extract_labels(df["id"])
        df = df[
            (df["coverage"] >= self.min_coverage) &
            (df["pident"] / 100 >= self.min_pident)
        ]
        return df

    def categorize_annotations(self, annotations: pd.Series) -> pd.Series:
        result = pd.Series("other_function", index=annotations.index)
        annotations = annotations.fillna("").astype(str)
        for category, pattern in self.compiled_patterns.items():
            mask = annotations.str.contains(pattern, na=False)
            result.loc[mask] = category
        return result

    def calculate_category_ratios(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> pd.DataFrame:
        df = df.copy()
        df["Category"] = self.categorize_annotations(df["Annotation"])
        counts = df.groupby(["id", "Category"]).size().unstack(fill_value=0)
        ratios = counts.div(counts.sum(axis=1), axis=0).round(4)
        ratios.columns = [
            f"{c.lower()}_ratio"
            for c in ratios.columns
        ]
        result = ratios.reset_index()
        if save_path:
            result.to_csv(save_path, sep=';', index=False)
        return result

    def process_file(self, in_root: Path, out_root: Path) -> pd.DataFrame:
        out_root.mkdir(parents=True, exist_ok=True)
        df = self.load_file(in_root)
        df = df.copy()
        filtered_df = self.preprocess(df)
        category_df = self.calculate_category_ratios(filtered_df)
        out_path = out_root / f"{in_root.stem[:3]}_PHV_FEAT.csv"
        category_df.to_csv(out_path, sep=';', index=False)
        return category_df
        