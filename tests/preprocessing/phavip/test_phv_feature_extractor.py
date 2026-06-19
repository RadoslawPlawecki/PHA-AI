"""
@author: Radosław Pławecki
"""

import pandas as pd
import pytest
from pathlib import Path
from preprocessing.phavip.feature_extractor import PhavipFeatureExtractor


@pytest.fixture(autouse=True)
def patch_format_accession(monkeypatch):
    import preprocessing.utils as utils
    monkeypatch.setattr(utils, "format_accession", lambda gtool_id, x: x)


@pytest.fixture
def extractor():
    return PhavipFeatureExtractor(min_coverage=0.7, min_pident=0.35)


def test_load_and_preprocess(tmp_path, extractor):
    input_file = tmp_path / "ABC_input.csv"
    pd.read_csv("project/tests/data/functional_annotation/input.csv", sep=";").to_csv(input_file, sep=";", index=False)
    df = extractor.load_file(input_file)
    df = extractor.preprocess(df)
    assert "label" in df.columns
    assert df["coverage"].min() >= 0.7
    assert (df["pident"] / 100).min() >= 0.35


def test_extract_labels():
    series = pd.Series(["x|S10", "y|S40", "z|S99"])
    labels = PhavipFeatureExtractor.extract_labels(series).tolist()
    assert labels == [1, 0, 0] 


def test_categorize_annotations(extractor):
    s = pd.Series([
        "capsid protein",
        "terminase enzyme",
        "holin protein",
        "unknown orf"
    ])
    out = extractor.categorize_annotations(s).tolist()
    assert "structural" in out
    assert "packaging" in out
    assert "lysis" in out
    assert "unknown" in out


def test_calculate_category_ratios(tmp_path, extractor):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "out.csv"
    pd.read_csv("project/tests/data/functional_annotation/input.csv", sep=";").to_csv(input_file, sep=";", index=False)
    df = extractor.load_file(input_file)
    df = extractor.preprocess(df)
    result = extractor.calculate_category_ratios(df, save_path=output_file)
    assert output_file.exists()
    saved = pd.read_csv(output_file, sep=";")
    assert not saved.empty
    pd.testing.assert_frame_equal(
        result.sort_index(axis=1),
        saved.sort_index(axis=1),
        check_exact=False,
        atol=1e-3
    )


def test_process_file(tmp_path, extractor):
    input_file = tmp_path / "ABC_input.csv"
    output_dir = tmp_path / "out"
    pd.read_csv("project/tests/data/functional_annotation/input.csv", sep=";").to_csv(input_file, sep=";", index=False)
    result = extractor.process_file(input_file, output_dir)
    assert (output_dir / "ABC_PHV_FEAT.csv").exists()
    assert isinstance(result, pd.DataFrame)
