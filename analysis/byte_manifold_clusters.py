#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
HEX_BASELINE_FILE = "hexagramdataset1.csv"
HEX_TUNNEL_FILE   = "tunneling_hexagram_dataset.csv"
BYTE_BASELINE_FILE = "byte_manifold_dataset.csv"
BYTE_TUNNEL_FILE   = "tunneling_byte_manifold_dataset.csv"

N_CLUSTERS_HEX  = 20
N_CLUSTERS_BYTE = 20

# ============================================================
# UTIL
# ============================================================
def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

# ============================================================
# HEXAGRAM MANIFOLD
# ============================================================
def load_hex_dataset(path):
    df = pd.read_csv(path)
    # hexagram_lines: "7-8-9-6-7-8"
    lines = df["hexagram_lines"].astype(str).str.split("-")
    X = np.array([[int(v) for v in row] for row in lines])
    # changing_lines count as curvature proxy
    cl = df["changing_lines"].astype(str)
    cl_count = cl.apply(lambda s: 0 if s == "none" else len(s.split("-"))).values
    return df, X, cl_count

# ============================================================
# BYTE MANIFOLD
# ============================================================
def load_byte_dataset(path):
    df = pd.read_csv(path)
    symbols = df["symbol_current"].astype(int).values
    X = np.array([[ (s >> i) & 1 for i in range(8) ] for s in symbols])
    bc = df["bit_change_count"].astype(int).values
    return df, X, bc

# ============================================================
# CLUSTERING + MARKOV + CURVATURE + TRANSITION ENTROPY
# ============================================================
def cluster_and_markov(X, curvature_vec, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_

    # cluster weights
    counts = np.bincount(labels, minlength=n_clusters)
    weights = counts / counts.sum()

    # cluster curvature (mean curvature per cluster)
    curv = np.zeros(n_clusters)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            curv[k] = 0.0
        else:
            curv[k] = curvature_vec[idx].mean()

    # Markov matrix over clusters
    T = np.zeros((n_clusters, n_clusters))
    for i in range(len(labels) - 1):
        a = labels[i]
        b = labels[i + 1]
        T[a, b] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    # transition entropy per cluster
    trans_entropy = np.array([shannon_entropy(P[i]) for i in range(n_clusters)])

    return {
        "labels": labels,
        "centers": centers,
        "weights": weights,
        "curvature": curv,
        "P": P,
        "stationary": stat,
        "transition_entropy": trans_entropy,
    }

# ============================================================
# DRIFT + BASIN PERSISTENCE
# ============================================================
def distribution_drift(a, b):
    L = max(len(a), len(b))
    a2 = np.pad(a, (0, L - len(a)))
    b2 = np.pad(b, (0, L - len(b)))
    return np.sum(np.abs(a2 - b2))

def basin_persistence(stat_a, stat_b, top_k=10):
    idx_a = np.argsort(stat_a)[::-1][:top_k]
    idx_b = np.argsort(stat_b)[::-1][:top_k]
    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    overlap = len(set_a & set_b)
    return {
        "top_a": idx_a,
        "top_b": idx_b,
        "overlap_count": overlap,
        "overlap_fraction": overlap / top_k,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    # ---------- load datasets ----------
    hex_base_df, hex_base_X, hex_base_curv = load_hex_dataset(HEX_BASELINE_FILE)
    hex_tun_df,  hex_tun_X,  hex_tun_curv  = load_hex_dataset(HEX_TUNNEL_FILE)

    byte_base_df, byte_base_X, byte_base_curv = load_byte_dataset(BYTE_BASELINE_FILE)
    byte_tun_df,  byte_tun_X,  byte_tun_curv  = load_byte_dataset(BYTE_TUNNEL_FILE)

    print("Loaded hex baseline:", len(hex_base_df))
    print("Loaded hex tunneling:", len(hex_tun_df))
    print("Loaded byte baseline:", len(byte_base_df))
    print("Loaded byte tunneling:", len(byte_tun_df))

    # ---------- analysis per manifold ----------
    hex_base_stats = cluster_and_markov(hex_base_X, hex_base_curv, N_CLUSTERS_HEX)
    hex_tun_stats  = cluster_and_markov(hex_tun_X,  hex_tun_curv,  N_CLUSTERS_HEX)

    byte_base_stats = cluster_and_markov(byte_base_X, byte_base_curv, N_CLUSTERS_BYTE)
    byte_tun_stats  = cluster_and_markov(byte_tun_X,  byte_tun_curv,  N_CLUSTERS_BYTE)

    # ---------- drift (stationary distributions) ----------
    hex_drift  = distribution_drift(hex_base_stats["stationary"],  hex_tun_stats["stationary"])
    byte_drift = distribution_drift(byte_base_stats["stationary"], byte_tun_stats["stationary"])

    # ---------- curvature drift (mean cluster curvature) ----------
    hex_curv_drift  = distribution_drift(hex_base_stats["curvature"],  hex_tun_stats["curvature"])
    byte_curv_drift = distribution_drift(byte_base_stats["curvature"], byte_tun_stats["curvature"])

    # ---------- transition entropy drift ----------
    hex_trans_drift  = distribution_drift(hex_base_stats["transition_entropy"],  hex_tun_stats["transition_entropy"])
    byte_trans_drift = distribution_drift(byte_base_stats["transition_entropy"], byte_tun_stats["transition_entropy"])

    # ---------- basin persistence ----------
    hex_basin = basin_persistence(hex_base_stats["stationary"], hex_tun_stats["stationary"], top_k=10)
    byte_basin = basin_persistence(byte_base_stats["stationary"], byte_tun_stats["stationary"], top_k=10)

    # ========================================================
    # OUTPUT
    # ========================================================
    print("\n=== Manifold-independent drift and curvature ===")
    print(f"hexagram: stationary_drift={hex_drift:.6f}  curvature_drift={hex_curv_drift:.6f}  trans_entropy_drift={hex_trans_drift:.6f}")
    print(f"byte:     stationary_drift={byte_drift:.6f} curvature_drift={byte_curv_drift:.6f} trans_entropy_drift={byte_trans_drift:.6f}")

    print("\n=== Basin persistence (top-10 clusters) ===")
    print("hexagram manifold:")
    print("  baseline top-10 clusters:", hex_basin["top_a"])
    print("  tunneling top-10 clusters:", hex_basin["top_b"])
    print(f"  overlap_count={hex_basin['overlap_count']}  overlap_fraction={hex_basin['overlap_fraction']:.3f}")

    print("byte manifold:")
    print("  baseline top-10 clusters:", byte_basin["top_a"])
    print("  tunneling top-10 clusters:", byte_basin["top_b"])
    print(f"  overlap_count={byte_basin['overlap_count']}  overlap_fraction={byte_basin['overlap_fraction']:.3f}")

    print("\n=== Cluster-level summaries (byte manifold, first 10 clusters) ===")
    for k in range(min(10, N_CLUSTERS_BYTE)):
        print(f"cluster {k}: weight_base={byte_base_stats['weights'][k]:.4f} "
              f"weight_tun={byte_tun_stats['weights'][k]:.4f} "
              f"curv_base={byte_base_stats['curvature'][k]:.4f} "
              f"curv_tun={byte_tun_stats['curvature'][k]:.4f} "
              f"H_base={byte_base_stats['transition_entropy'][k]:.4f} "
              f"H_tun={byte_tun_stats['transition_entropy'][k]:.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
