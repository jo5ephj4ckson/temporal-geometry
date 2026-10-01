import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("structured_output_clustered.csv")

# Extract hexagram lines as features
X = df[["line1", "line2", "line3", "line4", "line5", "line6"]].values

# ============================
# CLUSTERING METHODS
# ============================

# Standard K-Means
kmeans = KMeans(n_clusters=20, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# MiniBatch K-Means (memory safe)
mbk = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=1000)
labels_mbk = mbk.fit_predict(X)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=1.5, min_samples=10)
labels_dbscan = dbscan.fit_predict(X)

# ============================
# MARKOV CHAIN BUILDER
# ============================
def build_markov_chain(labels):
    unique = sorted(np.unique(labels))
    mapping = {label: i for i, label in enumerate(unique)}
    seq = np.array([mapping[l] for l in labels])
    n = len(unique)

    T = np.zeros((n, n), dtype=float)
    for i in range(len(seq) - 1):
        T[seq[i], seq[i+1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()

    return unique, stationary

# ============================
# RUN MARKOV CHAINS
# ============================
uniq_k, stat_k = build_markov_chain(labels_kmeans)
uniq_m, stat_m = build_markov_chain(labels_mbk)
uniq_d, stat_d = build_markov_chain(labels_dbscan)

# ============================
# ATTRACTOR PRINTER
# ============================
def print_attractors(name, uniq, stat, top=10):
    print("\n==============================")
    print(f"{name} ATTRACTORS")
    print("==============================")
    idxs = np.argsort(stat)[::-1][:top]
    for i in idxs:
        print(f"  State {uniq[i]}: {stat[i]:.6f}")

print_attractors("KMEANS", uniq_k, stat_k)
print_attractors("MINIBATCH KMEANS", uniq_m, stat_m)
print_attractors("DBSCAN", uniq_d, stat_d)

# ============================
# DRIFT BETWEEN METHODS
# ============================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

print("\n==============================")
print("CLUSTERING METHOD DRIFT")
print("==============================")
print(f"KMeans ↔ MiniBatchKMeans: {drift(stat_k, stat_m):.6f}")
print(f"KMeans ↔ DBSCAN:         {drift(stat_k, stat_d):.6f}")
print(f"MiniBatchKMeans ↔ DBSCAN:{drift(stat_m, stat_d):.6f}")

print("\nDone.")
