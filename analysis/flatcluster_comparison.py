import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_hp(path):
    print("\nLoading HP dataset:", path)
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

    print("Valid HP rows:", len(df))
    return pd.concat([df, parts.astype(int)], axis=1)

def load_alpha(path):
    print("\nLoading Alphabet dataset:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    if "symbol_id" not in df.columns:
        raise ValueError("Alphabet dataset missing required column: symbol_id")

    # Alphabet encoding is 1D, cluster directly on symbol_id
    X = df["symbol_id"].values.reshape(-1, 1)

    print("Valid Alphabet rows:", len(X))
    return X

def build_chain(X, k):
    print("Clustering with k =", k, "on", len(X), "samples...")
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    uniq = sorted(np.unique(labels))
    mapping = {u:i for i,u in enumerate(uniq)}
    seq = np.array([mapping[l] for l in labels])
    n = len(uniq)

    print("Actual clusters formed:", n)

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
# LOAD DATASETS
# ============================
hp_df = load_hp("hexagramdataset1.csv")
alpha_X = load_alpha("alphabetencoding.csv")

# HP uses 6D features
hp_X = hp_df[["line1","line2","line3","line4","line5","line6"]].values

# Pick safe k
k = min(20, len(hp_X), len(alpha_X))
print("\nUsing k =", k, "clusters.\n")

hp_stat, hp_strength, hp_k = build_chain(hp_X, k)
alpha_stat, alpha_strength, alpha_k = build_chain(alpha_X, k)

print("\n==============================")
print("ATTRACTOR / BASIN COMPARISON")
print("==============================")
print("Basin | HP_stat   Alpha_stat   HP_strength   Alpha_strength")
print("-----------------------------------------------------------")

for i in range(k):
    print(
        str(i).rjust(5),
        "|",
        f"{hp_stat[i]:8.6f}",
        f"{alpha_stat[i]:8.6f}",
        f"{hp_strength[i]:11.6f}",
        f"{alpha_strength[i]:11.6f}"
    )

print("\nDone.\n")
