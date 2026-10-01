import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOAD BOTH DATASETS
# ============================
df_rng = pd.read_csv("hexagram_dataset_with_rng.csv")
df_hp1 = pd.read_csv("hexagramdataset2.csv")

# ============================
# CLEAN DATA (DROP BAD ROWS)
# ============================
df_rng = df_rng.dropna(subset=["hexagram_lines"])
df_hp1 = df_hp1.dropna(subset=["hexagram_lines"])

# ============================
# SPLIT hexagram_lines INTO SIX COLUMNS
# ============================
def expand_hexagram_lines(df):
    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    # convert to integers, coercing bad values to NaN
    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    # drop rows where conversion failed
    parts = parts.dropna()

    # merge back with original df (aligned by index)
    df = df.loc[parts.index]
    return pd.concat([df, parts.astype(int)], axis=1)

df_rng = expand_hexagram_lines(df_rng)
df_hp1 = expand_hexagram_lines(df_hp1)

# Extract hexagram line vectors
cols = ["line1","line2","line3","line4","line5","line6"]
X_rng = df_rng[cols].values
X_hp1 = df_hp1[cols].values

# ============================
# CLUSTER BOTH WITH SAME MODEL
# ============================
k = 20

kmeans_rng = KMeans(n_clusters=k, random_state=42).fit(X_rng)
kmeans_hp1 = KMeans(n_clusters=k, random_state=42).fit(X_hp1)

labels_rng = kmeans_rng.labels_
labels_hp1 = kmeans_hp1.labels_

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

uniq_rng, stat_rng = build_markov(labels_rng)
uniq_hp1, stat_hp1 = build_markov(labels_hp1)

# ============================
# DRIFT MEASURE
# ============================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

drift_value = drift(stat_rng, stat_hp1)

print("\n==============================")
print("DELL vs HP1 CLUSTER SIMILARITY")
print("==============================")
print(f"Stationary distribution drift: {drift_value:.6f}")

print("\nTop attractors DELL:")
idxs_rng = np.argsort(stat_rng)[::-1][:10]
for i in idxs_rng:
    print(f"  State {uniq_rng[i]}: {stat_rng[i]:.6f}")

print("\nTop attractors HP1:")
idxs_hp1 = np.argsort(stat_hp1)[::-1][:10]
for i in idxs_hp1:
    print(f"  State {uniq_hp1[i]}: {stat_hp1[i]:.6f}")

print("\nDone.")
