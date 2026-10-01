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
    return parts.astype(int).values

def build_chain(labels, k):
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

    basin_self = np.diag(P)
    basin_outflow = P.sum(axis=1) / n
    strength = basin_self - basin_outflow

    return stat, strength

# ============================
# LOAD HP DATASET
# ============================
X = load_and_expand("hexagramdataset1.csv")

k = 20
print("\nUsing k =", k, "clusters.\n")

# ============================
# NORMAL CLUSTERING
# ============================
print("Clustering HP dataset normally...")
kmeans_normal = KMeans(n_clusters=k, random_state=42).fit(X)
labels_normal = kmeans_normal.labels_

normal_stat, normal_strength = build_chain(labels_normal, k)

# ============================
# HEXAGRAM-LINE RANDOMIZATION
# ============================
print("\nRandomizing hexagram lines inside each row...")

X_rand = np.copy(X)
for i in range(len(X_rand)):
    np.random.shuffle(X_rand[i])

print("Clustering randomized dataset...")
kmeans_rand = KMeans(n_clusters=k, random_state=42).fit(X_rand)
labels_rand = kmeans_rand.labels_

rand_stat, rand_strength = build_chain(labels_rand, k)

# ============================
# PRINT COMPARISON
# ============================
print("\n==============================")
print("HEXAGRAM-LINE RANDOMIZATION FALSIFICATION")
print("==============================")
print("Basin | Normal_stat   Rand_stat   Normal_strength   Rand_strength")
print("---------------------------------------------------------------")

for i in range(k):
    print(
        str(i).rjust(5),
        "|",
        f"{normal_stat[i]:8.6f}",
        f"{rand_stat[i]:8.6f}",
        f"{normal_strength[i]:11.6f}",
        f"{rand_strength[i]:11.6f}"
    )

print("\nDone.\n")
