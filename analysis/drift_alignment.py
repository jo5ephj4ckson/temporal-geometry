import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# GENERIC HEXAGRAM LOADER (HP, Dell, Random.org)
# ============================================================
def load_hexagram_dataset(path, label="Dataset"):
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
            raise ValueError(f"{label} missing line1–line6 columns.")

    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    print(f"Valid {label} rows:", len(parts))

    return parts.astype(int).values

# ============================================================
# ALPHABET LOADER
# ============================================================
def load_alphabet(path):
    print("\nLoading Alphabet dataset:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    if "symbol_id" not in df.columns:
        raise ValueError("Alphabet dataset missing symbol_id column.")

    X = df["symbol_id"].values.reshape(-1, 1)
    print("Valid Alphabet rows:", len(X))
    return X

# ============================================================
# STATIONARY DISTRIBUTION
# ============================================================
def build_stationary(X, k=20):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

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

    return stat

# ============================================================
# DRIFT METRIC
# ============================================================
def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

# ============================================================
# LOAD ALL SOURCES
# ============================================================
hp_X    = load_hexagram_dataset("hexagramdataset1.csv", "HP dataset")
alpha_X = load_alphabet("alphabetencoding.csv")
dell_X  = load_hexagram_dataset("hexagramdataset2.csv", "Dell dataset")
rand_X  = load_hexagram_dataset("randomdataset.csv", "Random.org dataset")

# ============================================================
# BUILD STATIONARY DISTRIBUTIONS
# ============================================================
print("\nComputing stationary distributions...")

hp_stat    = build_stationary(hp_X)
alpha_stat = build_stationary(alpha_X)
dell_stat  = build_stationary(dell_X)
rand_stat  = build_stationary(rand_X)

# ============================================================
# DRIFT ALIGNMENT OUTPUT
# ============================================================
print("\n==============================")
print("CROSS-SOURCE DRIFT ALIGNMENT")
print("==============================")

print(f"HP vs Alphabet:     {drift(hp_stat, alpha_stat):.6f}")
print(f"HP vs Dell:         {drift(hp_stat, dell_stat):.6f}")
print(f"HP vs Random.org:   {drift(hp_stat, rand_stat):.6f}")

print("\nDone.\n")
