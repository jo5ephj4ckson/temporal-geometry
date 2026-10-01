import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOAD DATASET
# ============================
df = pd.read_csv("hexagram_dataset_with_rng.csv")

# ============================
# CLEAN DATA
# ============================
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
# DYNAMIC WINDOW SIZE
# ============================
total = len(df)
window_size = total // 3   # one-third of dataset

early = df.iloc[:window_size]
late  = df.iloc[-window_size:]

cols = ["line1","line2","line3","line4","line5","line6"]
X_early = early[cols].values
X_late  = late[cols].values

# ============================
# CLUSTER BOTH WINDOWS
# ============================
k = 20

kmeans_early = KMeans(n_clusters=k, random_state=42).fit(X_early)
kmeans_late  = KMeans(n_clusters=k, random_state=42).fit(X_late)

labels_early = kmeans_early.labels_
labels_late  = kmeans_late.labels_

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

uniq_early, stat_early = build_markov(labels_early)
uniq_late,  stat_late  = build_markov(labels_late)

# ============================
# DRIFT MEASURE
# ============================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

drift_value = drift(stat_early, stat_late)

# ============================
# PRINT RESULTS
# ============================
print("\n==============================")
print("TEMPORAL STABILITY ANALYSIS")
print("==============================")
print(f"Rows in dataset: {total}")
print(f"Window size: {window_size}")
print(f"Stationary distribution drift (early vs late): {drift_value:.6f}")

print("\nTop attractors EARLY:")
idxs_e = np.argsort(stat_early)[::-1][:10]
for i in idxs_e:
    print(f"  State {uniq_early[i]}: {stat_early[i]:.6f}")

print("\nTop attractors LATE:")
idxs_l = np.argsort(stat_late)[::-1][:10]
for i in idxs_l:
    print(f"  State {uniq_late[i]}: {stat_late[i]:.6f}")

print("\nDone.")
