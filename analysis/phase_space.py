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

def reconstruct_phase_space(series, embedding_dim=3, lag=1):
    n = len(series)
    max_shift = (embedding_dim - 1) * lag
    n_eff = n - max_shift
    if n_eff <= 0:
        raise ValueError("Series too short for given embedding_dim and lag.")

    embedded = np.zeros((n_eff, embedding_dim), dtype=float)
    for i in range(embedding_dim):
        embedded[:, i] = series[i * lag : i * lag + n_eff]

    return embedded

# ============================
# LOAD DATASET
# ============================
X = load_and_expand("hexagramdataset1.csv")

k = 20
print("\nClustering into", k, "basins...")

kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

print("Basin sequence length:", len(labels))

# ============================
# PHASE SPACE RECONSTRUCTION
# ============================
embedding_dim = 3
lag = 1

print(f"\nReconstructing phase space with m={embedding_dim}, tau={lag}...")
points = reconstruct_phase_space(labels, embedding_dim=embedding_dim, lag=lag)

print("Phase-space points:", points.shape)

# ============================
# OUTPUT
# ============================
out_csv = "temporal_phase_space_points.csv"
out_txt = "temporal_phase_space_summary.txt"

pd.DataFrame(points, columns=[f"dim_{i}" for i in range(points.shape[1])]).to_csv(out_csv, index=False)

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("TEMPORAL PHASE-SPACE RECONSTRUCTION SUMMARY\n")
    f.write("==========================================\n\n")
    f.write(f"Number of points: {points.shape[0]}\n")
    f.write(f"Embedding dimension: {points.shape[1]}\n\n")
    f.write("Per-dimension statistics:\n")
    f.write("-------------------------\n")
    for i in range(points.shape[1]):
        dim = points[:, i]
        f.write(
            f"dim_{i}: mean={dim.mean():.6f} "
            f"std={dim.std(ddof=1):.6f} "
            f"min={dim.min():.6f} "
            f"max={dim.max():.6f}\n"
        )

print("\nDone. Output written to:")
print(out_csv)
print(out_txt)
