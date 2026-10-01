import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# UNIVERSAL LOADER FOR RANDOM.ORG STRUCTURED OUTPUT
# ============================================================

def load_structured_randomorg(path):
    df = pd.read_csv(path)

    # Expecting columns: line1–line6 (already expanded)
    line_cols = ["line1","line2","line3","line4","line5","line6"]
    if all(col in df.columns for col in line_cols):
        return df[line_cols].astype(int).values

    raise ValueError("structured_output_randomorg.csv must contain line1–line6 columns.")

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

    return stat, P, kmeans.cluster_centers_

# ============================================================
# BASIN STRENGTH CALCULATION
# ============================================================

def basin_strength(P):
    strengths = []
    for i in range(P.shape[0]):
        retention = P[i, i]
        outflow = (P[i].sum() - retention) / (P.shape[0] - 1)
        strengths.append(retention - outflow)
    return np.array(strengths)

# ============================================================
# MAIN EXECUTION
# ============================================================

path = "structured_output_randomorg.csv"
X = load_structured_randomorg(path)

k = 20

stat, P, centers = build_chain(X, k)
strengths = basin_strength(P)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================")
print(" RANDOM.ORG CLUSTER ANALYSIS")
print("==============================")
print(f"Samples: {len(X)}")
print(f"Clusters formed: {k}\n")

print("Basin | Stationary | Strength")
print("--------------------------------")
for i in range(k):
    print(
        str(i).rjust(5), "|",
        f"{stat[i]:10.6f}",
        f"{strengths[i]:10.6f}"
    )

print("\nCluster centers:")
print(centers)

print("\nDone.\n")
