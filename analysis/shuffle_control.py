import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOAD DATASET
# ============================
df = pd.read_csv("hexagram_dataset_with_rng.csv")
df = df.dropna(subset=["hexagram_lines"])

def expand_hexagram_lines(df):
    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]
    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")
    parts = parts.dropna()
    df = df.loc[parts.index]
    return pd.concat([df, parts.astype(int)], axis=1)

df = expand_hexagram_lines(df)

# ============================
# SHUFFLE CONTROL
# ============================
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

cols = ["line1","line2","line3","line4","line5","line6"]
X = df_shuffled[cols].values

# ============================
# CLUSTER
# ============================
k = 20
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

uniq = sorted(np.unique(labels))
mapping = {u:i for i,u in enumerate(uniq)}
seq = np.array([mapping[l] for l in labels])
n = len(uniq)
total = len(seq)

# ============================
# TRANSITION MATRIX
# ============================
T = np.zeros((n,n))
for i in range(len(seq)-1):
    T[seq[i], seq[i+1]] += 1

row_sums = T.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P = T / row_sums

# ============================
# STATIONARY DISTRIBUTION
# ============================
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigvals - 1))
stat = np.real(eigvecs[:, idx])
stat = stat / stat.sum()

# ============================
# BASIN METRICS
# ============================
basin_sizes = np.bincount(seq, minlength=n)
basin_density = basin_sizes / total
basin_inflow = P.sum(axis=0) / n
basin_outflow = P.sum(axis=1) / n
basin_self = np.diag(P)
basin_strength = basin_self - basin_outflow

# ============================
# PRINT RESULTS
# ============================
print("\n==============================")
print("SHUFFLE CONTROL ANALYSIS")
print("==============================")
print(f"Total samples: {total}\n")

print("Stationary distribution:")
for i in range(n):
    print(f"  Basin {uniq[i]}: {stat[i]:.6f}")

print("\nBasin strengths:")
for i in range(n):
    print(f"  Basin {uniq[i]}: {basin_strength[i]:.6f}")

print("\nDone.")
