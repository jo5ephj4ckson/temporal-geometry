import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# UNIVERSAL LOADER — works for BOTH hexagram and random datasets
# ============================================================

def load_hexagrams(path):
    df = pd.read_csv(path)

    # CASE 1: hexagram_lines exists (your hexagramdataset1.csv)
    if "hexagram_lines" in df.columns:
        parts = df["hexagram_lines"].str.split("-", expand=True)
        parts.columns = ["line1","line2","line3","line4","line5","line6"]
        parts = parts.astype(int)
        return parts.values

    # CASE 2: raw line1–line6 columns exist (your randomdataset.csv)
    line_cols = ["line1","line2","line3","line4","line5","line6"]
    if all(col in df.columns for col in line_cols):
        parts = df[line_cols].astype(int)
        return parts.values

    # CASE 3: nothing usable
    raise ValueError(
        f"{path} does not contain hexagram_lines or line1–line6 columns."
    )

# ============================================================
# BUILD MARKOV CHAIN + STATIONARY DISTRIBUTION
# ============================================================

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

    return stat

# ============================================================
# STATIONARY DISTRIBUTION DELTA
# ============================================================

def stationary_delta(hex_stat, rnd_stat):
    hex_stat = np.array(hex_stat)
    rnd_stat = np.array(rnd_stat)

    delta = hex_stat - rnd_stat
    abs_delta = np.abs(delta)
    total_drift = abs_delta.sum()

    print("\n==============================")
    print(" STATIONARY DISTRIBUTION DELTA")
    print("==============================")
    print("Basin | Hex_stat   Rnd_stat   Delta     AbsDelta")
    print("-----------------------------------------------")

    for i in range(len(hex_stat)):
        print(
            str(i).rjust(5), "|",
            f"{hex_stat[i]:10.6f}",
            f"{rnd_stat[i]:10.6f}",
            f"{delta[i]:10.6f}",
            f"{abs_delta[i]:10.6f}"
        )

    print("\nTotal drift magnitude:", total_drift)
    print("\nDone.\n")

# ============================================================
# MAIN EXECUTION
# ============================================================

hex_X = load_hexagrams("hexagramdataset1.csv")
rnd_X = load_hexagrams("randomdataset.csv")

k = 20

hex_stat = build_chain(hex_X, k)
rnd_stat = build_chain(rnd_X, k)

stationary_delta(hex_stat, rnd_stat)
