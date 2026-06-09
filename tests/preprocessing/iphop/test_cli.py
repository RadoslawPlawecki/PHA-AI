"""
@author: Radosław Pławecki
"""

from unittest.mock import Mock, patch
from preprocessing.iphop.cli import ask_feature_method, ask_normalization_method


def test_ask_feature_method():
    mock_prompt = Mock()
    mock_prompt.ask.return_value = "1) Predation Pressure [pp]"
    with patch("preprocessing.iphop.cli.questionary.select", return_value=mock_prompt):
        result = ask_feature_method()
    assert result == "1) Predation Pressure [pp]"


def test_ask_normalization_method():
    mock_prompt = Mock()
    mock_prompt.ask.return_value = "2) CLR + Z-score [clr_z]"
    with patch("preprocessing.iphop.cli.questionary.select", return_value=mock_prompt):
        result = ask_normalization_method()
    assert result == "2) CLR + Z-score [clr_z]"
