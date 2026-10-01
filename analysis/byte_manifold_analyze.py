#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG — set your dataset filenames here
# ============================================================
BASELINE_FILE = "byte_manifold_dataset.csv"
TUNNEL_FILE   = "tunneling_byte_manifold_dataset.csv"

# ============================================================
# LOAD DATA
# ============================================================
def load_byte_dataset(path):
    df = pd.read_csv(path)
    # symbol_current is 0–255
    # convert to 8-bit vector
    symbols = df["symbol_current"].astype(int).values
    X = np.array([[ (s >> i) & 1 for i in range(8) ] for s in symbols])
    return df, X

df_base, X_base = load_byte_dataset(BASELINE_FILE)
df_tunnel, X_tunnel = load_byte_dataset(TUNNEL_FILE)

print("Loaded baseline:", len(df_base))
print("Loaded tunneling:", len(df_tunnel))

# ============================================================
# CLUSTERING
# ============================================================
K = 20  # number of clusters (same as hexagram analysis)

kmeans_base = KMeans(n_clusters=K, random_state=42).fit(X_base)
kmeans_tunnel = KMeans(n_clusters=K, random_state=42).fit(X_tunnel)

labels_base = kmeans_base.labels_
labels_tunnel = kmeans_tunnel.labels_

# ============================================================
# MARKOV CHAIN BUILDER
# ============================================================
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

    # stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    return uniq, stat

uniq_base, stat_base = build_markov(labels_base)
uniq_tunnel, stat_tunnel = build_markov(labels_tunnel)

# ============================================================
# DRIFT MEASURE
# ============================================================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

drift_value = drift(stat_base, stat_tunnel)

# ============================================================
# OUTPUT
# ============================================================
print("\n==============================")
print("BYTE MANIFOLD CLUSTER / MARKOV / DRIFT")
print("==============================")
print(f"Stationary distribution drift: {drift_value:.6f}")

print("\nTop attractors (baseline RNG):")
idxs_base = np.argsort(stat_base)[::-1][:10]
for i in idxs_base:
    print(f"  State {uniq_base[i]}: {stat_base[i]:.6f}")

print("\nTop attractors (tunneling RNG):")
idxs_tunnel = np.argsort(stat_tunnel)[::-1][:10]
for i in idxs_tunnel:
    print(f"  State {uniq_tunnel[i]}: {stat_tunnel[i]:.6f}")

print("\nDone.")
