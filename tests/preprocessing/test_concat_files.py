"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path
from preprocessing.concat_files import merge_tsvs


def _create_tsv(path: Path, data: dict):
    df = pd.DataFrame(data)
    df.to_csv(path, sep="\t", index=False)


def test_merge_tsvs(tmp_path):
    in_root = tmp_path / "raw"
    out_root = tmp_path / "raw-merged"
    gtool_dir = in_root / "toolA" / "gtool1"
    gtool_dir.mkdir(parents=True, exist_ok=True)
    _create_tsv(gtool_dir / "S10_gtool_ChV_ftool.tsv", {
        "a": [1, 2],
        "b": [3, 4]
    })
    _create_tsv(gtool_dir / "S11_gtool_ChV_ftool.tsv", {
        "a": [5],
        "b": [6]
    })
    merge_tsvs(in_root, out_root)
    out_file = out_root / "toolA" / "gtool_ChV_ftool_M.csv"
    assert out_file.exists(), f"Missing output file: {out_file}"
    df = pd.read_csv(out_file, sep=";")
    assert len(df) == 3
    assert set(df.columns) == {"a", "b"}
    assert df["a"].tolist() == [1, 2, 5]
    assert df["b"].tolist() == [3, 4, 6]


def test_empty_gtool(tmp_path):
    in_root = tmp_path / "raw"
    out_root = tmp_path / "merged"
    (in_root / "toolA" / "g1").mkdir(parents=True)
    merge_tsvs(in_root, out_root)
    assert not (out_root / "toolA").exists()


def test_ignores_non_directories(tmp_path):
    in_root = tmp_path / "raw"
    out_root = tmp_path / "merged"
    tool_dir = in_root / "toolA"
    tool_dir.mkdir(parents=True)
    (tool_dir / "not_a_dir.txt").write_text("ignore me")
    merge_tsvs(in_root, out_root)  


def test_multiple_ptools(tmp_path):
    in_root = tmp_path / "raw"
    out_root = tmp_path / "merged"
    for tool in ["A", "B"]:
        g = in_root / f"tool{tool}" / "g1"
        g.mkdir(parents=True)
        df = pd.DataFrame({"x": [1]})
        df.to_csv(g / f"{tool}_sample.tsv", sep="\t", index=False)
    merge_tsvs(in_root, out_root)
    assert (out_root / "toolA").exists()
    assert (out_root / "toolB").exists()


def test_ignores_non_dirs(tmp_path):
    in_root = tmp_path / "raw"
    in_root.mkdir()
    (in_root / "not_a_dir.txt").write_text("x")
    merge_tsvs(in_root, tmp_path / "out")
