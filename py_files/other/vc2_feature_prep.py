"""
@author: Radosław Pławecki
"""

import pandas as pd
import os


class vC2FeaturePrep:
    def __init__(self, input_path, min_patients=2, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        if min_patients > 0:
            self.min_patients = min_patients
        else:
            print("[ERROR] Minimum number of patients with VC should be more than 0.")
            return None
        self.allowed_statuses = ['Clustered', 'Clustered/Singleton']
        
    def run(self):
        if not os.path.exists(self.input_path):
            print(f"[ERROR] Input file not found: {self.input_path}.")
            return None

        df = pd.read_csv(self.input_path)

        df.columns = ['Genome', 'Order', 'Family', 'Genus', 'preVC', 'Status', 'VC'] + list(df.columns[7:])
        print(f"\n[INFO] {self.input_path}")
        print(f"[INFO] {len(df)} rows loaded.")

        df_filtered = df[df['Status'].isin(self.allowed_statuses)].copy()
        df_filtered = df_filtered[df_filtered['Genome'].str.match(r'^S\d+_', na=False)]

        if df_filtered.empty:
            print("[WARNING] No records after filtering.")
            return None

        df_filtered['id'] = df_filtered['Genome'].str.split('_').str[0]

        binary_matrix = pd.crosstab(df_filtered['VC'], df_filtered['id'])
        binary_matrix = (binary_matrix > 0).astype(int)

        vc_mask = binary_matrix.sum(axis=1) >= self.min_patients
        df_out = binary_matrix[vc_mask].T

        sample_id = df_out.index.str.replace('S', '').astype(int)
        label = (sample_id <= 34).astype(int)

        df_out.insert(0, 'label', label)

        if self.output_path:
            df_out.to_csv(self.output_path)
            print(f"[SUCCESS] Saved to {self.output_path}")

        return df_out
