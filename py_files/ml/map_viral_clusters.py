"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np


ROOTS = [
    'data/vcontact2/output/genomad/geN_ChV_PHA_PROT_vC2/genome_by_genome_overview.csv',
    'data/vcontact2/output/virsorter2/VS2_ChV_PHA_PROT_vC2/genome_by_genome_overview.csv',
    'data/vcontact2/output/vibrant/VIB_ChV_PHA_PROT_vC2/genome_by_genome_overview.csv',
]

OUTPUTS = [
    'data/ml/geN_vC2.csv',
    'data/ml/VS2_vC2.csv',
    'data/ml/VIB_vC2.csv',
]


allowed_statuses = ['Clustered', 'Clustered/Singleton']
min_patients = 2

for file_path, out_path in zip(ROOTS, OUTPUTS):
    df = pd.read_csv(file_path, delimiter=',')

    df.columns = ['Genome', 'Order', 'Family', 'Genus', 'preVC', 'Status', 'VC'] + list(df.columns[7:])

    print(f"\n[INFO] {file_path}")
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

    # remove rare clusters
    vc_mask = binary_matrix.sum(axis=1) >= min_patients

    # transpose matrix to Patient x VC
    df = binary_matrix[vc_mask].T

    # add labels
    sample_id = df.index.str.replace('S', '').astype(int)
    label = (sample_id <= 34).astype(int)

    df.insert(0, 'label', label)
    print(f"[INFO] matrix {df.shape} generated.")

    # save per tool
    df.to_csv(out_path)
