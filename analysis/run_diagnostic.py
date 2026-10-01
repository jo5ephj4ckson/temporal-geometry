import pandas as pd

df1 = pd.read_csv("hexagramdataset1.csv")
df2 = pd.read_csv("hexagramdataset2.csv")

print("DATASET 1 COLUMNS:")
print(df1.columns)

print("\nDATASET 2 COLUMNS:")
print(df2.columns)
