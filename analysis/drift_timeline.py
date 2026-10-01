import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOADING FUNCTIONS
# ============================

def load_hp(path):
    df = pd.read_csv(path)
    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]
    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")
    parts = parts.dropna()
    return parts.astype(int).values

def load_alpha(path):
    df = pd.read_csv(path)
    X = df["symbol_id"].values.reshape(-1, 1)
    return X

# ============================
# STATIONARY DISTRIBUTION
# ============================

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

# ============================
# DRIFT METRIC
# ============================

def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

# ============================
# MAIN: DRIFT OVER TIME
# ============================

hp_X = load_hp("hexagramdataset1.csv")
alpha_X = load_alpha("alphabetencoding.csv")

max_len = min(len(hp_X), len(alpha_X))
step = 10000

timeline = []

for N in range(step, max_len, step):
    hp_slice = hp_X[:N]
    alpha_slice = alpha_X[:N]

    hp_stat = build_stationary(hp_slice)
    alpha_stat = build_stationary(alpha_slice)

    d = drift(hp_stat, alpha_stat)

    timeline.append((N, d))
    print(f"Samples: {N}, Drift: {d:.6f}")

print("\n==============================")
print("HP1 vs Alphabet Drift Timeline")
print("==============================")
for N, d in timeline:
    print(f"{N} samples -> drift {d:.6f}")
