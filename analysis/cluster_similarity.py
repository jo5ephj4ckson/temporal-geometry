import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOAD BOTH DATASETS
# ============================
df1 = pd.read_csv("hexagramdataset1.csv")
df2 = pd.read_csv("hexagramdataset2.csv")

# ============================
# CLEAN + PARSE hexagram_lines
# ============================
def clean_and_parse(df):
    # Remove rows where hexagram_lines is missing or contains nan-like text
    df = df[df["hexagram_lines"].notna()]
    df = df[~df["hexagram_lines"].str.lower().isin(["nan", "none", ""])]
    
    # Split into 6 integers
    parsed = df["hexagram_lines"].str.split("-", expand=True)
    parsed = parsed.astype(int)
    return parsed.values

X1 = clean_and_parse(df1)
X2 = clean_and_parse(df2)

# ============================
# CLUSTER BOTH WITH SAME MODEL
# ============================
k = 20

kmeans1 = KMeans(n_clusters=k, random_state=42).fit(X1)
kmeans2 = KMeans(n_clusters=k, random_state=42).fit(X2)

labels1 = kmeans1.labels_
labels2 = kmeans2.labels_

# ============================
# MARKOV CHAIN BUILDER
# ============================
def build_markov(labels):
    uniq = sorted(np.unique(labels))
    mapping = {u:i for i,u in enumerate(uniq)}
    seq = np.array([mapping[l] for l in labels])
    n = len(uniq)

    T = np.zeros((n,n))
    for i in range(len(seq)-1):
        T[seq[i], seq[i+1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    return uniq, stat

uniq1, stat1 = build_markov(labels1)
uniq2, stat2 = build_markov(labels2)

# ============================
# DRIFT MEASURE
# ============================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

drift_value = drift(stat1, stat2)

print("\n==============================")
print("HP1 vs HP2 CLUSTER SIMILARITY")
print("==============================")
print(f"Stationary distribution drift: {drift_value:.6f}")

print("\nTop attractors HP1:")
idxs1 = np.argsort(stat1)[::-1][:10]
for i in idxs1:
    print(f"  State {uniq1[i]}: {stat1[i]:.6f}")

print("\nTop attractors HP2:")
idxs2 = np.argsort(stat2)[::-1][:10]
for i in idxs2:
    print(f"  State {uniq2[i]}: {stat2[i]:.6f}")

print("\nDone.")
