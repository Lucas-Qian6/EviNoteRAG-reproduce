import pandas as pd

# read data
df = pd.read_parquet('./data_preprocess/data/m_test.parquet')

# select the first data
row = df.iloc[0]

for col in df.columns:
    print(f"name: {col}")
    print(f"content: {row[col]}\n{'='*40}")
    print()
