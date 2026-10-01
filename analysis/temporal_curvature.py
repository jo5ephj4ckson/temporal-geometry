import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# UNIVERSAL LOADER — auto-detect timestamp + numeric columns
# ============================================================

def load_dataset(path):
    print(f"\n--- Loading dataset: {path} ---")

    df = pd.read_csv(path)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    # ---- Detect timestamp column ----
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if len(ts_candidates) > 0:
        ts_col = ts_candidates[0]
        print(f"Timestamp column detected: {ts_col}")
        timestamps = df[ts_col].astype(str).values
    else:
        print("No timestamp column detected — using row index as time.")
        timestamps = np.arange(len(df)).astype(str)

    # ---- Hexagram lines ----
    if "hexagram_lines" in df.columns:
        print("Detected hexagram_lines column.")
        parts = df["hexagram_lines"].str.split("-", expand=True)
        parts = parts.apply(pd.to_numeric, errors="coerce").dropna()
        print(f"Hexagram numeric shape: {parts.shape}")
        return parts.values, timestamps

    # ---- Alphabet encoding ----
    if "symbol_id" in df.columns:
        print("Detected symbol_id column.")
        X = df["symbol_id"].apply(pd.to_numeric, errors="coerce").dropna().values.reshape(-1,1)
        print(f"Alphabet numeric shape: {X.shape}")
        return X, timestamps

    # ---- Random.org numeric entropy ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        print(f"Detected numeric entropy column: {col}")
        X = df[col].values.reshape(-1,1)
        print(f"Numeric entropy shape: {X.shape}")
        return X, timestamps

    print("ERROR: No usable numeric symbolic columns found.")
    raise ValueError(f"No usable numeric symbolic columns found in {path}")


# ============================================================
# STATIONARY DISTRIBUTION + TEMPORAL CURVATURE
# ============================================================

def build_stationary_and_curvature(X, timestamps, k=20):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_

    uniq = sorted(np.unique(labels))
    mapping = {u:i for i,u in enumerate(uniq)}
    seq = np.array([mapping[l] for l in labels])
    n = len(uniq)

    # ---- Transition matrix ----
    T = np.zeros((n,n))
    for i in range(len(seq)-1):
        T[seq[i], seq[i+1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # ---- Stationary distribution ----
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    # ---- Temporal curvature ----
    expected_self = 1.0 / n
    actual_self = np.diag(P)

    local_curvature = actual_self - expected_self
    global_curvature = np.var(local_curvature)

    return stat, P, local_curvature, global_curvature


# ============================================================
# DRIFT METRIC
# ============================================================

def drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))


# ============================================================
# DATASETS — YOUR FILENAMES EXACTLY
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


# ============================================================
# MAIN LOOP
# ============================================================

k_values = [12, 20, 32]
results = []
curvature_results = []

for k in k_values:
    print(f"\n==============================")
    print(f"Temporal invariance + curvature at k = {k}")
    print("==============================")

    stats = {}
    global_curv = {}

    for name, (X, ts) in loaded.items():
        stat, P, local_curv, g_curv = build_stationary_and_curvature(X, ts, k=k)
        stats[name] = stat
        global_curv[name] = g_curv
        curvature_results.append((k, name, g_curv))
        print(f"{name}: stationary length {len(stat)}, global curvature {g_curv:.6f}")

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


# ============================================================
# SUMMARY OUTPUT
# ============================================================

print("\n==============================")
print("Temporal curvature summary")
print("==============================")
for k, name, g_curv in curvature_results:
    print(f"k={k:2d} | {name:9s} -> global curvature {g_curv:.6f}")

print("\n==============================")
print("Temporal invariance drift summary")
print("==============================")
for k, a, b, d in results:
    print(f"k={k:2d} | {a:9s} vs {b:9s} -> drift {d:.6f}")
