"""
@author: Radosław Pławecki
The script to move the directories that failed while being analysed with CheckV.
"""

import os
import shutil

data_path = "./data/checkv/virsorter2"
target_path = "./data/spoiled"

os.makedirs(target_path, exist_ok=True)

for directory in os.listdir(data_path):
    dir_path = os.path.join(data_path, directory)

    if not os.path.isdir(dir_path):
        continue

    files = os.listdir(dir_path)

    if files == ["tmp"]:
        print(f"Moving: {dir_path}")

        shutil.move(
            dir_path,
            os.path.join(target_path, directory)
        )
