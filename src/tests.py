import pandas as pd
df = pd.read_csv("data/processed/clean_dataset.csv")
print(df.columns.tolist())
print(len(df.columns))
