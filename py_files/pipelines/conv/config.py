from dataclasses import dataclass


@dataclass
class Config:
    in_file = "data/phabox2/richness/PGN_R.csv"
    output_dir = "data/results/conv/phagcn"
    tools = [
        "genomad",
        "virsorter2",
        "vibrant"
    ]
    alpha = 0.05
