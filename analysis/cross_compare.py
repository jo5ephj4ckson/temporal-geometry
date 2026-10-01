import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# ============================================================
# LOAD DATASETS
# ============================================================

android = pd.read_csv("android_hexagrams.csv")
hp1 = pd.read_csv("hexagramdataset1.csv")
tunnel = pd.read_csv("tunneling_hexagram_dataset.csv")

# ============================================================
# PARSE HEXAGRAM STRING "7-8-7-7-7-6" → [7,8,7,7,7,6]
# ============================================================

def parse_hexagram_string(s):
    if isinstance(s, str):
        return list(map(int, s.split("-")))
    return [0,0,0,0,0,0]  # fallback

def extract_android_vectors(df):
    return df[["line1","line2","line3","line4","line5","line6"]].values

def extract_hexagram_vectors(df):
    return np.array([parse_hexagram_string(x) for x in df["hexagram_lines"]])

X_android = extract_android_vectors(android)
X_hp1 = extract_hexagram_vectors(hp1)
X_tunnel = extract_hexagram_vectors(tunnel)

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

K = 4

def run_kmeans(X):
    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    return km.cluster_centers_, labels

android_centers, android_labels = run_kmeans(X_android)
hp1_centers, hp1_labels = run_kmeans(X_hp1)
tunnel_centers, tunnel_labels = run_kmeans(X_tunnel)

print("\n=== CLUSTER CENTERS: ANDROID ===")
print(android_centers)

print("\n=== CLUSTER CENTERS: HP1 ===")
print(hp1_centers)

print("\n=== CLUSTER CENTERS: TUNNELING ===")
print(tunnel_centers)

# ============================================================
# CLUSTER COUNTS
# ============================================================

def cluster_counts(labels):
    return np.bincount(labels, minlength=K)

print("\n=== CLUSTER COUNTS ===")
print("ANDROID:", cluster_counts(android_labels))
print("HP1:", cluster_counts(hp1_labels))
print("TUNNEL:", cluster_counts(tunnel_labels))

# ============================================================
# TRANSITION MATRICES
# ============================================================

def transition_matrix(labels):
    M = np.zeros((K, K), dtype=int)
    for i in range(len(labels)-1):
        M[labels[i], labels[i+1]] += 1
    return M

print("\n=== TRANSITION MATRIX: ANDROID ===")
print(transition_matrix(android_labels))

print("\n=== TRANSITION MATRIX: HP1 ===")
print(transition_matrix(hp1_labels))

print("\n=== TRANSITION MATRIX: TUNNELING ===")
print(transition_matrix(tunnel_labels))

# ============================================================
# CLUSTER CENTER DIVERGENCE
# ============================================================

def center_divergence(A, B):
    return pairwise_distances(A, B)

print("\n=== CENTER DIVERGENCE (ANDROID vs HP1) ===")
print(center_divergence(android_centers, hp1_centers))

print("\n=== CENTER DIVERGENCE (ANDROID vs TUNNELING) ===")
print(center_divergence(android_centers, tunnel_centers))

print("\n=== CENTER DIVERGENCE (HP1 vs TUNNELING) ===")
print(center_divergence(hp1_centers, tunnel_centers))

# ============================================================
# KL DIVERGENCE FOR CLUSTER COUNTS
# ============================================================

def normalize(v):
    v = v.astype(float)
    return v / v.sum()

def kl_divergence(p, q):
    p = normalize(p)
    q = normalize(q)
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))

android_c = cluster_counts(android_labels)
hp1_c = cluster_counts(hp1_labels)
tunnel_c = cluster_counts(tunnel_labels)

print("\n=== KL DIVERGENCE (ANDROID vs HP1) ===")
print(kl_divergence(android_c, hp1_c))

print("\n=== KL DIVERGENCE (ANDROID vs TUNNELING) ===")
print(kl_divergence(android_c, tunnel_c))

print("\n=== KL DIVERGENCE (HP1 vs TUNNELING) ===")
print(kl_divergence(hp1_c, tunnel_c))
