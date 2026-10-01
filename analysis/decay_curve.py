import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_and_expand(path):
    print("\nLoading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

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

def compute_lag_correlations(labels, max_lag=10):
    n = len(labels)
    k = len(np.unique(labels))

    onehot = np.zeros((n, k))
    for i, lab in enumerate(labels):
        onehot[i, lab] = 1

    lag_results = {}

    for lag in range(1, max_lag+1):
        A = onehot[:-lag]
        B = onehot[lag:]

        C = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                C[i, j] = np.corrcoef(A[:, i], B[:, j])[0, 1]

        lag_results[lag] = C

    return lag_results

def compute_decay_curve(lag_results):
    decay = []
    for lag, C in lag_results.items():
        # Use mean absolute correlation as the decay metric
        mean_abs_corr = np.mean(np.abs(C))
        decay.append((lag, mean_abs_corr))
    return decay

# ============================
# LOAD HP DATASET
# ============================
X = load_and_expand("hexagramdataset1.csv")

k = 20
print("\nUsing k =", k, "clusters.\n")

# ============================
# CLUSTER HP DATASET
# ============================
print("Clustering HP dataset...")
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

# ============================
# COMPUTE LAG CORRELATIONS
# ============================
max_lag = 10
print(f"Computing lag correlations up to lag {max_lag}...")
lag_results = compute_lag_correlations(labels, max_lag=max_lag)

# ============================
# COMPUTE DECAY CURVE
# ============================
decay_curve = compute_decay_curve(lag_results)

# ============================
# OUTPUT RESULTS
# ============================
output_path = "temporal_memory_decay_curve.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("TEMPORAL MEMORY DECAY CURVE\n")
    f.write("===========================\n\n")
    f.write("Lag | MeanAbsCorrelation\n")
    f.write("------------------------\n")

    for lag, val in decay_curve:
        f.write(f"{lag:3d} | {val:.8f}\n")

print(f"\nDone. Output written to {output_path}\n")
