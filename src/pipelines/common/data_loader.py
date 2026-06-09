"""
@author: Radosław Pławecki
"""

import pandas as pd
import questionary
import os


class DataLoader:
    AVAILABLE_TOOLS = {"vcontact2", "cherry", "phagcn", "phavip"}

    def __init__(
        self, 
        input_path: str, 
        min_patients: int = 2, 
        tool: str | None = None, 
        output_path: str | None = None
    ):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}.")
        if tool not in self.AVAILABLE_TOOLS:
            raise ValueError(
                f"Unknown data from tool: {tool}. "
                f"Available tools: {', '.join(self.AVAILABLE_TOOLS)}"
            )
        if min_patients <= 0:
            raise ValueError("min_patients must be greater than 0.")
        
        self.input_path = input_path
        self.output_path = output_path
        self.tool = tool
        self.min_patients = min_patients

    def process(self, col=None):
        """Method to route to the specific tool processor and return features/labels."""
        if self.tool == "vcontact2":
            df = self._process_vcontact2()
        else:
            df = self._process_generic(col)

        sample_ids = df.index.astype(str).str.extract(r'\|S(\d+)')[0]
        sample_ids = pd.to_numeric(sample_ids, errors='coerce')
        
        y = (sample_ids <= 34).astype(int).values
        X = df.drop(columns=['label'], errors='ignore')
        
        return X.values, y, X.columns

    def _process_vcontact2(self) -> pd.DataFrame:
        """Method to handle processing for vContact2 data."""
        df = pd.read_csv(self.input_path)
        self._log_load(df)
        
        df.columns = ['Genome', 'Order', 'Family', 'Genus', 'preVC', 'Status', 'VC'] + list(df.columns[7:])
        
        allowed_statuses = ['Clustered', 'Clustered/Singleton']
        df_filtered = df[df['Status'].isin(allowed_statuses)].copy()
        df_filtered = df_filtered[df_filtered['Genome'].str.match(r'^[^|]+\|S\d+_', na=False)]
        
        if df_filtered.empty:
            raise ValueError("No records after filtering.")
            
        df_filtered['id'] = df_filtered['Genome'].str.split('_').str[0]
        
        return self._build_matrix(df_filtered, feature_col='VC', id_col='id')

    def _process_generic(self, col=None) -> pd.DataFrame:
        """Method to handle processing for tools with identical CSV structures (Cherry, PhageGCN)."""
        df = pd.read_csv(self.input_path, delimiter=';')
        self._log_load(df)
        
        df = df[df['Accession'].str.match(r'^[^|]+\|S\d+_', na=False)].copy()
        df['id'] = df['Accession'].str.split('_').str[0]
        
        if col is None:
            selectable_columns = [col for col in df.columns if col not in {'Accession', 'id'}]
            if not selectable_columns:
                raise ValueError("No valid columns to choose from.")
                
            col = questionary.select(
                "Choose a column:",
                choices=selectable_columns
            ).ask()
            print(f"\n[INFO] Selected column: {col}")
            
        return self._build_matrix(df, feature_col=col, id_col='id')

    def _build_matrix(self, df: pd.DataFrame, feature_col: str, id_col: str) -> pd.DataFrame:
        """Method to generate the binary cross-tabulation matrix and save it if necessary."""
        binary_matrix = pd.crosstab(df[feature_col], df[id_col])
        binary_matrix = (binary_matrix > 0).astype(int)
        
        vc_mask = binary_matrix.sum(axis=1) >= self.min_patients
        df_out = binary_matrix[vc_mask].T
        
        if self.output_path:
            df_out.to_csv(self.output_path)
            print(f"[SUCCESS] Saved to {self.output_path}")
            
        return df_out

    def _log_load(self, df: pd.DataFrame):
        """Method for consistent logging."""
        print(f"\n[INFO] {self.input_path}")
        print(f"[INFO] {len(df)} rows loaded.")