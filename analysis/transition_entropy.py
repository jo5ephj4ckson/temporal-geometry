import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_and_expand(path):
    print("\nLoading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    # Build hexagram_lines if missing
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

def build_transition_matrix(labels, k):
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
    return P

def shannon_entropy(prob_row):
    # Remove zeros to avoid log issues
    p = prob_row[prob_row > 0]
    return -np.sum(p * np.log2(p))

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
# BUILD TRANSITION MATRIX
# ============================
print("Building transition matrix...")
P = build_transition_matrix(labels, k)

# ============================
# COMPUTE ENTROPY PER BASIN
# ============================
print("Computing basin transition entropy...")

entropy_values = []
for i in range(k):
    H = shannon_entropy(P[i])
    entropy_values.append(H)

# ============================
# OUTPUT RESULTS TO FILE
# ============================
output_path = "basin_transition_entropy.txt"
with open(output_path, "w") as f:
    f.write("BASIN TRANSITION ENTROPY\n")
    f.write("========================\n\n")
    f.write("Basin | Entropy (bits)\n")
    f.write("----------------------\n")

    for i, H in enumerate(entropy_values):
        f.write(f"{i:5d} | {H:.6f}\n")

print(f"\nDone. Output written to {output_path}\n")
