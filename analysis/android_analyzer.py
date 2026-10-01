import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("android_hexagrams.csv")

# Extract the 6-line vectors
X = df[["line1","line2","line3","line4","line5","line6"]].values
timestamps = df["timestamp"].values

# ============================================================
# K-MEANS CLUSTERING
# ============================================================
K = 4  # adjust as needed
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
k_labels = kmeans.fit_predict(X)

print("\n=== K-MEANS CLUSTER CENTERS ===")
print(kmeans.cluster_centers_)

print("\n=== K-MEANS CLUSTER COUNTS ===")
for i in range(K):
    print(f"Cluster {i}: {np.sum(k_labels == i)} samples")

# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================
H = 4  # adjust as needed
agg = AgglomerativeClustering(n_clusters=H)
h_labels = agg.fit_predict(X)

print("\n=== AGGLOMERATIVE CLUSTER COUNTS ===")
for i in range(H):
    print(f"Cluster {i}: {np.sum(h_labels == i)} samples")

# ============================================================
# MARKOV TRANSITION MATRIX (K-MEANS)
# ============================================================
print("\n=== K-MEANS TRANSITION MATRIX ===")
transition_matrix = np.zeros((K, K), dtype=int)

for i in range(len(k_labels) - 1):
    a = k_labels[i]
    b = k_labels[i+1]
    transition_matrix[a, b] += 1

print(transition_matrix)

# ============================================================
# MARKOV TRANSITION MATRIX (AGGLOMERATIVE)
# ============================================================
print("\n=== AGGLOMERATIVE TRANSITION MATRIX ===")
transition_matrix_h = np.zeros((H, H), dtype=int)

for i in range(len(h_labels) - 1):
    a = h_labels[i]
    b = h_labels[i+1]
    transition_matrix_h[a, b] += 1

print(transition_matrix_h)

# ============================================================
# DRIFT PATH (ORDERED BY TIMESTAMP)
# ============================================================
print("\n=== DRIFT PATH (K-MEANS) ===")
ordered = np.argsort(timestamps)
ordered_labels = k_labels[ordered]

print(ordered_labels.tolist())
