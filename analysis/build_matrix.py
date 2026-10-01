import pandas as pd
import numpy as np

# === Load HDBSCAN output ===
df = pd.read_csv("hdbscan_output.csv")

# === Sort by time to ensure correct sequence ===
df = df.sort_values("timestamp_unix").reset_index(drop=True)

# === Compute transitions ===
transitions = []
for i in range(len(df) - 1):
    current_cluster = df.loc[i, "hdbscan_cluster"]
    next_cluster = df.loc[i + 1, "hdbscan_cluster"]
    transitions.append((current_cluster, next_cluster))

# === Build transition matrix ===
unique_clusters = sorted(df["hdbscan_cluster"].unique())
matrix = pd.DataFrame(0, index=unique_clusters, columns=unique_clusters)

for c1, c2 in transitions:
    matrix.loc[c1, c2] += 1

# === Normalize to probabilities ===
matrix_prob = matrix.div(matrix.sum(axis=1), axis=0)

print("Transition counts:")
print(matrix)
print("\nTransition probabilities:")
print(matrix_prob)
