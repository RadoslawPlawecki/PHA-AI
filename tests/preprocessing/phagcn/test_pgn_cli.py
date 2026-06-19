"""
@author: Radosław Pławecki
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch
from preprocessing.phagcn.cli import ask_column


def test_ask_column():
    df = pd.DataFrame({
        "genus": [1, 2],
        "species": [3, 4],
        "Accession": [5, 6],
        "id": [7, 8],
    })
    mock_prompt = Mock()
    mock_prompt.ask.return_value = "genus"
    with patch("preprocessing.phagcn.cli.questionary.select", return_value=mock_prompt):
        result = ask_column(df)
    assert result == "genus"


def test_ask_column_raises_value_error():
    df = pd.DataFrame({
        "Accession": [1, 2],
        "id": [3, 4],
    })
    with pytest.raises(ValueError, match="No valid columns to choose from"):
        ask_column(df)
