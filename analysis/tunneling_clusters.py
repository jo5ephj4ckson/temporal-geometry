import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# LOAD + EXPAND HEXAGRAM DATA
# ============================================================

def load_and_expand(path):
    print("\nLoading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    # Build hexagram_lines if missing
    if "hexagram_lines" not in df.columns:
        needed = ["line1","line2","line3","line4","line5","line6"]
        if all(col in df.columns for col in needed):
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

    # Expand hexagram_lines
    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    df = df.loc[parts.index]

    print("Valid rows after expansion:", len(df))
    return pd.concat([df, parts.astype(int)], axis=1)

# ============================================================
# CLUSTERING FUNCTION
# ============================================================

def build_clusters(df, k):
    cols = ["line1","line2","line3","line4","line5","line6"]
    X = df[cols].values

    print("\nClustering with k =", k, "on", len(X), "samples...")
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

    return stat, strength, n, P

# ============================================================
# MAIN
# ============================================================

hex_df = load_and_expand("hexagramdataset1.csv")
tun_df = load_and_expand("tunneling_hexagram_dataset.csv")

# Pick safe k
k = min(20, len(hex_df), len(tun_df))
print("\nUsing k =", k, "clusters.\n")

hex_stat, hex_strength, hex_k, hex_P = build_clusters(hex_df, k)
tun_stat, tun_strength, tun_k, tun_P = build_clusters(tun_df, k)

# ============================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================

print("\n==============================")
print("CLUSTER COMPARISON")
print("==============================")
print("Cluster | Hex_stat   Tun_stat   Hex_strength   Tun_strength")
print("-----------------------------------------------------------")

for i in range(k):
    print(
        str(i).rjust(7),
        "|",
        f"{hex_stat[i]:8.6f}",
        f"{tun_stat[i]:8.6f}",
        f"{hex_strength[i]:11.6f}",
        f"{tun_strength[i]:11.6f}"
    )

print("\nDone.\n")
