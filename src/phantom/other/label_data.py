"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np


def classify(val):
    if isinstance(val, str) and val.startswith('S'):
        try:
            num = int(val.split('_')[0][1:])
            if num <= 34:
                return 'Allergy'
            else: 
                return 'Healthy'
        except:
            return 'Other'
    return 'Other'


df = pd.read_csv("genome_by_genome_overview.csv", delimiter=',')
df['Group'] = df['Genome'].apply(classify)

result = df[['Genome', 'Group']]

result.to_csv('mapping.csv', index=False, sep=';')
