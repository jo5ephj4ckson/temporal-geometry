import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_and_expand(path):
    print("\nLoading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    # Build hexagram_lines if missing
    if "hexagram_lines" not in df.columns:
        df["hexagram_lines"] = (
            df["line1"].astype(str) + "-" +
            df["line2"].astype(str) + "-" +
            df["line3"].astype(str) + "-" +
            df["line4"].astype(str) + "-" +
            df["line5"].astype(str) + "-" +
            df["line6"].astype(str)
        )

    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    df = df.loc[parts.index]

    print("Valid rows:", len(df))
    return pd.concat([df, parts.astype(int)], axis=1)

def build_chain(labels, k):
    uniq = sorted(np.unique(labels))
    mapping = {u:i for i,u in enumerate(uniq)}
    seq = np.array([mapping[l] for l in labels])
    n = len(uniq)

    # Transition matrix
    T = np.zeros((n,n))
    for i in range(len(seq)-1):
        T[seq[i], seq[i+1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    basin_self = np.diag(P)
    basin_outflow = P.sum(axis=1) / n
    strength = basin_self - basin_outflow

    return stat, strength, n

# ============================
# LOAD HP DATASET
# ============================
hp_df = load_and_expand("hexagramdataset1.csv")
hp_X = hp_df[["line1","line2","line3","line4","line5","line6"]].values

# ============================
# GENERATE SYNTHETIC RANDOM DATA
# ============================
print("\nGenerating synthetic random hexagrams...")
num_samples = len(hp_X)
rand_X = np.random.randint(0, 2, size=(num_samples, 6))

print("Synthetic random rows:", len(rand_X))

k = 20
print("\nUsing k =", k, "clusters.\n")

# ============================
# CLUSTER HP DATA
# ============================
print("Clustering HP dataset...")
kmeans_hp = KMeans(n_clusters=k, random_state=42).fit(hp_X)
labels_hp = kmeans_hp.labels_

hp_stat, hp_strength, _ = build_chain(labels_hp, k)

# ============================
# CLUSTER SYNTHETIC RANDOM DATA
# ============================
print("\nClustering synthetic random dataset...")
kmeans_rand = KMeans(n_clusters=k, random_state=42).fit(rand_X)
labels_rand = kmeans_rand.labels_

rand_stat, rand_strength, _ = build_chain(labels_rand, k)

# ============================
# PRINT COMPARISON
# ============================
print("\n==============================")
print("SYNTHETIC RANDOM FALSIFICATION")
print("==============================")
print("Basin | HP_stat   Rand_stat   HP_strength   Rand_strength")
print("---------------------------------------------------------")

for i in range(k):
    print(
        str(i).rjust(5),
        "|",
        f"{hp_stat[i]:8.6f}",
        f"{rand_stat[i]:8.6f}",
        f"{hp_strength[i]:11.6f}",
        f"{rand_strength[i]:11.6f}"
    )

print("\nDone.\n")
