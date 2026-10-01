import pandas as pd

df = pd.read_csv("hexagram_dataset_with_rng.csv")
df = df.dropna(subset=["hexagram_lines"])
print("Rows after cleaning:", len(df))
