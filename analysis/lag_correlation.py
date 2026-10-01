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

    # One-hot encode basin labels
    onehot = np.zeros((n, k))
    for i, lab in enumerate(labels):
        onehot[i, lab] = 1

    lag_results = {}

    for lag in range(1, max_lag+1):
        # Align sequences
        A = onehot[:-lag]
        B = onehot[lag:]

        # Compute correlation matrix
        C = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                C[i, j] = np.corrcoef(A[:, i], B[:, j])[0, 1]

        lag_results[lag] = C

    return lag_results

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
print(f"Computing temporal-lag correlations up to lag {max_lag}...")

lag_results = compute_lag_correlations(labels, max_lag=max_lag)

# ============================
# OUTPUT RESULTS
# ============================
output_path = "temporal_lag_correlation_map.txt"
with open(output_path, "w") as f:
    f.write("TEMPORAL-LAG CORRELATION MAP\n")
    f.write("============================\n\n")

    for lag, C in lag_results.items():
        f.write(f"\n--- Lag = {lag} ---\n")
        for i in range(k):
            row = " ".join([f"{C[i,j]: .4f}" for j in range(k)])
            f.write(f"Basin {i:2d}: {row}\n")

print(f"\nDone. Output written to {output_path}\n")
