import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_hexagrams(path):
    df = pd.read_csv(path)

    # Build hexagram_lines if missing
    if "hexagram_lines" not in df.columns:
        cols = ["line1","line2","line3","line4","line5","line6"]
        if all(c in df.columns for c in cols):
            df["hexagram_lines"] = (
                df["line1"].astype(str) + "-" +
                df["line2"].astype(str) + "-" +
                df["line3"].astype(str) + "-" +
                df["line4"].astype(str) + "-" +
                df["line5"].astype(str) + "-" +
                df["line6"].astype(str)
            )
        else:
            raise ValueError("Missing line1–line6 columns.")

    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    df = df.loc[parts.index]

    return parts.astype(int)

def build_chain(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

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

    return stat, strength, n, kmeans.cluster_centers_

def analyze_growth(path, k=20):
    X = load_hexagrams(path).values
    N = len(X)

    print(f"\nDataset size: {N} samples")
    print(f"Using k = {min(k, N)} clusters")

    stat, strength, n, centers = build_chain(X, min(k, N))

    print("\n==============================")
    print(" GROWTH ANALYSIS (HEXAGRAMS) ")
    print("==============================")
    print("Basin | Stationary | Strength")
    print("--------------------------------")

    for i in range(n):
        print(
            str(i).rjust(5),
            "|",
            f"{stat[i]:10.6f}",
            f"{strength[i]:10.6f}"
        )

    print("\nCluster centers:")
    print(centers)

    print("\nDone.\n")

# Run it
analyze_growth("hexagramdataset1.csv", k=20)
