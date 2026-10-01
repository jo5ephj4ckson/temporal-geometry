import pandas as pd

df = pd.read_csv("structured_output.csv")  # or whatever file you’re actually using
print(df.columns.tolist())
