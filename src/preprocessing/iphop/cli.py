"""
@author: Radosław Pławecki
"""

import questionary


def ask_feature_method() -> str:
    method = questionary.select(
        "Select feature engineering method for the phage-host relation:",
        choices=[
            "1) Predation Pressure [pp]",
            "2) Occurrence Matrix [om]",
        ],
    ).ask()
    print(f"\n[INFO] Selected feature engineering method: {method}")
    return method


def ask_normalization_method() -> str:
    method = questionary.select(
        "Select normalization method:",
        choices=[
            "1) TSS + Z-score [tss_z]",
            "2) CLR + Z-score [clr_z]",
            "3) Only TSS [tss]",
            "4) Nothing (raw data) [raw]",
        ],
    ).ask()
    print(f"[INFO] Selected normalization: {method}")
    return method
