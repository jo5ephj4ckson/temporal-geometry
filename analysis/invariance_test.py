import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# UNIVERSAL LOADER — auto-detect timestamp + numeric columns
# ============================================================

def load_dataset(path):
    df = pd.read_csv(path)

    # ---- Detect timestamp column ----
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if len(ts_candidates) > 0:
        timestamps = df[ts_candidates[0]].astype(str).values
    else:
        # fallback: preserve row order even without timestamps
        timestamps = np.arange(len(df)).astype(str)

    # ---- Detect numeric symbolic columns ----
    # Priority 1: hexagram_lines (split into 6 numeric columns)
    if "hexagram_lines" in df.columns:
        parts = df["hexagram_lines"].str.split("-", expand=True)
        parts = parts.apply(pd.to_numeric, errors="coerce").dropna()
        X = parts.values
        return X, timestamps

    # Priority 2: symbol_id (alphabetic encoding)
    if "symbol_id" in df.columns:
        X = df["symbol_id"].apply(pd.to_numeric, errors="coerce").dropna().values.reshape(-1,1)
        return X, timestamps

    # Priority 3: any numeric entropy column (Random.org)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X = df[numeric_cols[0]].values.reshape(-1,1)
        return X, timestamps

    raise ValueError(f"No usable numeric symbolic columns found in {path}")


# ============================================================
# STATIONARY DISTRIBUTION
# ============================================================

def build_stationary(X, timestamps, k=20):
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
# MAIN: TEMPORAL INVARIANCE ACROSS k
# ============================================================

datasets = {
    "HP1": "hexagramdataset1.csv",
    "Dell": "hexagram_dataset_with_rng.csv",
    "Alphabet": "flat64_dataset.csv",
    "RandomOrg": "structured_output_randomorg.csv",
    "Inverted": "inverted_hexagram_dataset.csv",
}

loaded = {}
for name, path in datasets.items():
    X, ts = load_dataset(path)
    loaded[name] = (X, ts)

k_values = [12, 20, 32]
results = []

for k in k_values:
    print(f"\n==============================")
    print(f"Temporal invariance analysis at k = {k}")
    print("==============================")

    stats = {}
    for name, (X, ts) in loaded.items():
        stat = build_stationary(X, ts, k=k)
        stats[name] = stat
        print(f"{name}: stationary length {len(stat)}")

    pairs = [
        ("HP1", "Dell"),
        ("HP1", "Alphabet"),
        ("HP1", "RandomOrg"),
        ("HP1", "Inverted"),
        ("Dell", "Alphabet"),
        ("Dell", "RandomOrg"),
        ("Alphabet", "RandomOrg"),
        ("Alphabet", "Inverted"),
        ("RandomOrg", "Inverted"),
    ]

    for a, b in pairs:
        d = drift(stats[a], stats[b])
        results.append((k, a, b, d))
        print(f"Drift({a} vs {b}) at k={k}: {d:.6f}")

print("\n==============================")
print("Temporal invariance drift summary")
print("==============================")
for k, a, b, d in results:
    print(f"k={k:2d} | {a:9s} vs {b:9s} -> drift {d:.6f}")
