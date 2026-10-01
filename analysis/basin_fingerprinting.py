import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def load_hexagram_dataset(path, label):
    print(f"\nLoading {label}: {path}")
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    # Build hexagram_lines if missing
    if "hexagram_lines" not in df.columns:
        if all(col in df.columns for col in ["line1","line2","line3","line4","line5","line6"]):
            df["hexagram_lines"] = (
                df["line1"].astype(str) + "-" +
                df["line2"].astype(str) + "-" +
                df["line3"].astype(str) + "-" +
                df["line4"].astype(str) + "-" +
                df["line5"].astype(str) + "-" +
                df["line6"].astype(str)
            )
        else:
            raise ValueError(f"{label} missing line1-line6 columns.")

    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    print(f"Valid {label} rows:", len(parts))

    return parts.astype(int).values

def cluster_and_centroids(X, k=20):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def fingerprint(centroids_A, centroids_B):
    D = cdist(centroids_A, centroids_B, metric="euclidean")
    matches = np.argmin(D, axis=1)
    distances = np.min(D, axis=1)
    return matches, distances

# ============================
# LOAD ALL SOURCES
# ============================
hp_X    = load_hexagram_dataset("hexagramdataset1.csv", "HP dataset")
dell_X  = load_hexagram_dataset("hexagramdataset2.csv", "Dell dataset")
rand_X  = load_hexagram_dataset("randomdataset.csv", "Random.org dataset")

k = 20
print(f"\nUsing k = {k} clusters.\n")

# ============================
# CLUSTER EACH SOURCE
# ============================
print("Clustering HP...")
hp_labels, hp_centroids = cluster_and_centroids(hp_X, k)

print("Clustering Dell...")
dell_labels, dell_centroids = cluster_and_centroids(dell_X, k)

print("Clustering Random.org...")
rand_labels, rand_centroids = cluster_and_centroids(rand_X, k)

# ============================
# FINGERPRINT MATCHING
# ============================
print("\nComputing cross-hardware basin fingerprints...\n")

hp_to_dell_match, hp_to_dell_dist = fingerprint(hp_centroids, dell_centroids)
hp_to_rand_match, hp_to_rand_dist = fingerprint(hp_centroids, rand_centroids)

# ============================
# OUTPUT RESULTS (UTF-8 SAFE)
# ============================
output_path = "basin_fingerprints_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("CROSS-HARDWARE BASIN FINGERPRINTS\n")
    f.write("=================================\n\n")
    f.write("Format: HP_basin -> closest Dell_basin (distance)\n")
    f.write("        HP_basin -> closest Random_basin (distance)\n\n")

    for i in range(k):
        f.write(
            f"HP {i:2d} -> Dell {hp_to_dell_match[i]:2d}   dist={hp_to_dell_dist[i]:.4f}   "
            f"| Random {hp_to_rand_match[i]:2d}   dist={hp_to_rand_dist[i]:.4f}\n"
        )

print(f"\nDone. Output written to {output_path}\n")
