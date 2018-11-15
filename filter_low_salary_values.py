import pandas as pd
import pdb

df = pd.read_csv('data/cleaned_data.csv')

df = df[ (df.min_salary > 40000) & (df.max_salary > 40000) ]


df.to_csv('data/low_salaries_removed.csv')
