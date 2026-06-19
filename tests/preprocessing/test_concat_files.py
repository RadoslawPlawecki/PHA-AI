"""
@author: Radosław Pławecki
"""

import pandas as pd
from pathlib import Path
from preprocessing.concat_files import merge_tsvs, merge_csvs


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


def make_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, sep=";")


def test_merge_single_subdir_multiple_csvs(tmp_path, capsys):
    subdir = tmp_path / "data"
    subdir.mkdir()
    make_csv(subdir / "a_part1.csv", [{"x": 1}, {"x": 2}])
    make_csv(subdir / "b_part2.csv", [{"x": 3}])
    merge_csvs(tmp_path)
    out_files = list(subdir.glob("ALL_*.csv"))
    assert len(out_files) == 1
    out_file = out_files[0]
    df = pd.read_csv(out_file, sep=";")
    assert len(df) == 3
    assert list(df["x"]) == [1, 2, 3]
    captured = capsys.readouterr().out
    assert "Merged 2 files" in captured


def test_merge_skips_non_directory(tmp_path):
    (tmp_path / "ignore.csv").write_text("x;y\n1;2\n")
    merge_csvs(tmp_path)
    assert list(tmp_path.glob("ALL_*.csv")) == []


def test_merge_skips_empty_subdir(tmp_path):
    (tmp_path / "empty").mkdir()
    merge_csvs(tmp_path)
    assert list((tmp_path / "empty").glob("ALL_*.csv")) == []


def test_output_name_logic(tmp_path):
    subdir = tmp_path / "data"
    subdir.mkdir()
    make_csv(subdir / "prefix_alpha.csv", [{"a": 1}])
    make_csv(subdir / "prefix_beta.csv", [{"a": 2}])
    merge_csvs(tmp_path)
    out_file = next(subdir.glob("ALL_*.csv"))
    assert out_file.name == "ALL_alpha.csv"


def test_files_are_sorted_before_merge(tmp_path):
    subdir = tmp_path / "data"
    subdir.mkdir()
    make_csv(subdir / "z_second.csv", [{"v": 2}])
    make_csv(subdir / "a_first.csv", [{"v": 1}])
    merge_csvs(tmp_path)
    out_file = next(subdir.glob("ALL_*.csv"))
    df = pd.read_csv(out_file, sep=";")
    assert list(df["v"]) == [1, 2]


def test_multiple_subdirs(tmp_path):
    sub1 = tmp_path / "d1"
    sub2 = tmp_path / "d2"
    sub1.mkdir()
    sub2.mkdir()
    make_csv(sub1 / "p_one.csv", [{"x": 1}])
    make_csv(sub2 / "p_two.csv", [{"y": 2}])
    merge_csvs(tmp_path)
    assert len(list(sub1.glob("ALL_*.csv"))) == 1
    assert len(list(sub2.glob("ALL_*.csv"))) == 1
