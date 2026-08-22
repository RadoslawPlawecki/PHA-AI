"""
Track A: full-dataset search. Same data picks and scores the winner, so 
the result is optimistic.
"""

import json
import os
import pandas as pd
from ..search_core import RunConfig, SEARCH_FNS


def run_exploratory(tool: str, df: pd.DataFrame, config: RunConfig, out_dir, file_path=None):
    study = SEARCH_FNS[tool](
        df, model_name=config.model, validator_name=config.validator,
        target_metric=config.metric, use_smote=config.smote, file_path=file_path,
    )
    exp_dir = os.path.join(out_dir, "exploratory")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "exploratory_result.json"), "w") as f:
        json.dump({
            "tool": tool,
            "target_metric": config.metric,
            "best_score": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2, default=str)
    return study
