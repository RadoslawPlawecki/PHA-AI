"""
Per-tool feature pipeline for PhaVIP (functional annotation) output:
cleaning/filtering (used by the preprocessing step) and feature-matrix
construction (used by the extraction step).
"""

import pandas as pd
import re

from ..matrix import derive_patient_id


class PhavipFeaturePipeline:
    # Derives its own functional categories from Annotation text; needs no
    # interactively chosen column (see build_feature_matrix).
    NEEDS_FEATURE_COLUMN = False
    # Feature ratios are fixed per category, not a binary/count matrix
    # over a min-patient cutoff (see build_feature_matrix).
    NEEDS_MATRIX_OPTIONS = False

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

    # categorize_annotations() falls back to "other_function" for anything
    # unmatched by CATEGORY_PATTERNS, so the true output category set has
    # one more member than CATEGORY_PATTERNS itself.
    ALL_CATEGORIES = list(CATEGORY_PATTERNS) + ["other_function"]

    def __init__(self, min_coverage: float = 0.7, min_pident: float = 0.35):
        self.min_coverage = min_coverage
        self.min_pident = min_pident
        self.compiled_patterns = {
            k: re.compile(v, re.IGNORECASE)
            for k, v in self.CATEGORY_PATTERNS.items()
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df["pident"] = pd.to_numeric(df["pident"], errors="coerce")
        df = df[
            (df["coverage"] >= self.min_coverage) &
            (df["pident"] / 100 >= self.min_pident)
        ]
        if "Category" not in df.columns:
            df["Category"] = self.categorize_annotations(df["Annotation"])
        return df[["Accession", "Annotation", "Category"]]

    def categorize_annotations(self, annotations: pd.Series) -> pd.Series:
        result = pd.Series("other_function", index=annotations.index)
        annotations = annotations.fillna("").astype(str)
        for category, pattern in self.compiled_patterns.items():
            mask = annotations.str.contains(pattern, na=False)
            result.loc[mask] = category
        return result

    def build_feature_matrix(
        self, df: pd.DataFrame, feature_col: str | None = None,
        binary: bool | None = None, min_patients: int | None = None,
        feature_columns: list[str] | None = None
    ) -> pd.DataFrame:
        # feature_col/binary/min_patients are unused: categories are fixed
        # (see CATEGORY_PATTERNS), not interactively chosen. Kept for a
        # uniform pipeline interface.
        df = df.copy()
        df["id"] = derive_patient_id(df["Accession"])
        counts = df.groupby(["id", "Category"]).size().unstack(fill_value=0)
        # Reindex to the full fixed category set: a patient subset (e.g. one
        # outer-CV fold) can easily have zero rows in some categories, which
        # would otherwise silently drop that column from just this subset's
        # matrix.
        counts = counts.reindex(columns=self.ALL_CATEGORIES, fill_value=0)
        ratios = counts.div(counts.sum(axis=1), axis=0).round(4)
        ratios.columns = [f"{c.lower()}_ratio" for c in ratios.columns]
        if feature_columns is not None:
            ratios = ratios.reindex(columns=feature_columns, fill_value=0)
        return ratios.reset_index()
