"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np
import os

roots_dict = {
    "genomad": "geN",
    "virsorter2": "VS2",
    "vibrant": "VIB"
}

input_path = 'data/vcontact2/output/'
output_path = 'data/ml/'

allowed_statuses = ['Clustered', 'Clustered/Singleton']

roots = os.listdir(input_path)
for root in roots:
    for i in range(1, 11):
        min_patients = i

        in_path = os.path.join(input_path, root, f'{roots_dict[root]}_ChV_PHA_PROT_vC2/genome_by_genome_overview.csv')

        df = pd.read_csv(in_path, delimiter=',')

        df.columns = ['Genome', 'Order', 'Family', 'Genus', 'preVC', 'Status', 'VC'] + list(df.columns[7:])

        print(f"\n[INFO] {in_path}")
        print(f"[INFO] {len(df)} rows loaded.")

        # only clustered data
        df_filtered = df[df['Status'].isin(allowed_statuses)].copy()

        # remove reference viruses (S = reference)
        df_filtered = df_filtered[df_filtered['Genome'].str.match(r'^S\d+_', na=False)]

        # patient id
        df_filtered['id'] = df_filtered['Genome'].str.split('_').str[0]

        print(f"[INFO] {len(df_filtered)} rows after filtering.")

        # VC x Patient matrix
        binary_matrix = pd.crosstab(df_filtered['VC'], df_filtered['id'])

        # binarization
        binary_matrix = (binary_matrix > 0).astype(int)

        # dimensionality reduction
        vc_mask = binary_matrix.sum(axis=1) >= min_patients

        # transpose matrix to Patient x VC
        df = binary_matrix[vc_mask].T

        # add labels
        sample_id = df.index.str.replace('S', '').astype(int)
        label = (sample_id <= 34).astype(int)

        df.insert(0, 'label', label)
        print(f"[INFO] matrix {df.shape} generated.")
        
        # save per tool
        dir_path = os.path.join(output_path, root)
        os.makedirs(dir_path, exist_ok=True)

        out_path = os.path.join(dir_path, f'{roots_dict[root]}_vC2_mP{min_patients}.csv')
        df.to_csv(out_path)
        print(f"[INFO] output written to: {out_path}")
