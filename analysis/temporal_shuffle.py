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
X = hp_df[["line1","line2","line3","line4","line5","line6"]].values

k = 20
print("\nUsing k =", k, "clusters.\n")

# ============================
# NORMAL CLUSTERING (UNSHUFFLED)
# ============================
print("Clustering HP dataset normally...")
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels_normal = kmeans.labels_

normal_stat, normal_strength, _ = build_chain(labels_normal, k)

# ============================
# TEMPORAL SHUFFLE
# ============================
print("\nShuffling temporal order...")
shuffle_idx = np.random.permutation(len(labels_normal))
labels_shuffled = labels_normal[shuffle_idx]

shuffle_stat, shuffle_strength, _ = build_chain(labels_shuffled, k)

# ============================
# PRINT COMPARISON
# ============================
print("\n==============================")
print("TEMPORAL SHUFFLE FALSIFICATION")
print("==============================")
print("Basin | Normal_stat   Shuffled_stat   Normal_strength   Shuffled_strength")
print("-------------------------------------------------------------------------")

for i in range(k):
    print(
        str(i).rjust(5),
        "|",
        f"{normal_stat[i]:8.6f}",
        f"{shuffle_stat[i]:8.6f}",
        f"{normal_strength[i]:11.6f}",
        f"{shuffle_strength[i]:11.6f}"
    )

print("\nDone.\n")
