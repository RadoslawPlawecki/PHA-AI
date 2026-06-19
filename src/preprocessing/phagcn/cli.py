"""
@author: Radosław Pławecki
"""

import questionary
import pandas as pd


def ask_column(df: pd.DataFrame) -> str:
    selectable_columns = [col for col in df.columns if col not in {'Accession', 'id'}]
    if not selectable_columns:
        raise ValueError("No valid columns to choose from.")
    col = questionary.select(
        "Choose a column:",
        choices=selectable_columns
    ).ask()
    print(f"\n[INFO] Selected column: {col}")
    return col
