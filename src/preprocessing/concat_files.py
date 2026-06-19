"""
@author: Radosław Pławecki
"""

from pathlib import Path
import pandas as pd


def merge_tsvs(in_root: Path, out_root: Path):
    """Merge tsv files into one."""
    for ptool_dir in in_root.iterdir():
        if not ptool_dir.is_dir():
            continue
        for gtool_dir in ptool_dir.iterdir():
            if not gtool_dir.is_dir():
                continue
            dfs = []
            sample_name = None
            for file in sorted(gtool_dir.glob("*.tsv")):
                if sample_name is None:
                    parts = file.stem.split('_')[1:]
                    sample_name = "_".join(parts)
                df = pd.read_csv(file, sep="\t")
                dfs.append(df)
            if not dfs:
                continue
            merged = pd.concat(dfs, ignore_index=True)
            out_dir = out_root / ptool_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sample_name}_M.csv"
            merged.to_csv(out_path, sep=';', index=False)
            print(f"\nPTool: {ptool_dir.name} | GTool: {gtool_dir.name}")
            print(merged.shape)
            print(merged.head())


def merge_csvs(in_root: str | Path):
    """Merge csv files into one."""
    in_root = Path(in_root)
    for subdir in in_root.iterdir():
        if not subdir.is_dir():
            continue
        csv_files = sorted(subdir.glob("*.csv"))
        if not csv_files:
            continue
        merged_df = pd.concat(
            (
                pd.read_csv(f, delimiter=';', on_bad_lines="skip")
                for f in csv_files
            ),
            ignore_index=True
        )
        output_name = f"ALL_{csv_files[0].stem.split('_', 1)[1]}.csv"
        print(output_name)
        output_path = subdir / output_name
        merged_df.to_csv(output_path, index=False, sep=';')
        print(f"Merged {len(csv_files)} files -> {output_path}")


"""in_root = Path("data/modalities/raw")
out_root = Path("data/modalities/raw-merged")
merge_tsvs(in_root=in_root, out_root=out_root)"""
# merge_csvs("data/modalities/raw-merged")
