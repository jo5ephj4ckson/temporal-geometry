import pandas as pd
import numpy as np
import hdbscan

df = pd.read_csv("structured_output.csv")

# Parse hexagram lines
hex_cols = ["line1", "line2", "line3", "line4", "line5", "line6"]
df[hex_cols] = df[hex_cols].astype(int)

# Parse changing_lines
def parse_changing(x):
    x = str(x).strip().lower()
    if x in ["", "none", "nan"]:
        return 0
    return len(x.split("-"))

df["num_changing"] = df["changing_lines"].apply(parse_changing)

# Build feature matrix
X = df[hex_cols + ["num_changing"]].values

# Cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
labels = clusterer.fit_predict(X)

df["cluster"] = labels

df.to_csv("structured_output_clustered.csv", index=False)

print("Clusters:", sorted(df["cluster"].unique()))
