#!/usr/bin/env python3
import csv
import math
import numpy as np

# ============================================================
# CONFIG — USE YOUR REAL FILENAMES HERE
# ============================================================

HEX_BASELINE_FILE   = "hexagramdataset1.csv"
HEX_TUNNEL_FILE     = "tunneling_hexagram_dataset.csv"
BYTE_BASELINE_FILE  = "byte_manifold_dataset.csv"
BYTE_TUNNEL_FILE    = "tunneling_byte_manifold_dataset.csv"

N_CLUSTERS = 20


# ============================================================
# LOADERS
# ============================================================

def load_hex_dataset(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r["hexagram_lines"])
    return rows

def load_byte_dataset(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(int(r["symbol_current"]))
    return rows


# ============================================================
# STATE ENCODING
# ============================================================

def encode_states(seq):
    mapping = {}
    encoded = []
    for s in seq:
        if s not in mapping:
            mapping[s] = len(mapping)
        encoded.append(mapping[s])
    return np.array(encoded), mapping


# ============================================================
# MARKOV CHAIN
# ============================================================

def build_markov(states, n_states):
    M = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        a = states[i]
        b = states[i + 1]
        M[a, b] += 1

    for i in range(n_states):
        s = M[i].sum()
        if s > 0:
            M[i] /= s
        else:
            M[i] = 1.0 / n_states

    return M


def stationary_distribution(P, tol=1e-10):
    n = P.shape[0]
    pi = np.ones(n) / n
    while True:
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi, 1) < tol:
            return new_pi
        pi = new_pi


# ============================================================
# METRICS
# ============================================================

def curvature(pi):
    return -np.log(pi + 1e-12)

def conditional_entropy(P):
    H = 0.0
    for row in P:
        for p in row:
            if p > 0:
                H += -p * math.log2(p)
    return H / P.shape[0]

def drift(a, b):
    return float(np.linalg.norm(a - b, 1))


# ============================================================
# CLUSTERING (1D k-means on stationary distribution)
# ============================================================

def kmeans(values, k):
    centers = np.linspace(values.min(), values.max(), k)
    labels = np.zeros(len(values), dtype=int)

    for _ in range(100):
        for i, v in enumerate(values):
            labels[i] = np.argmin(np.abs(centers - v))

        new_centers = centers.copy()
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                new_centers[c] = values[idx].mean()

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return labels, centers


def cluster_weights(pi, labels, k):
    w = np.zeros(k)
    for i, p in enumerate(pi):
        w[labels[i]] += p
    return w


def cluster_curvature(curv, labels, k):
    out = np.zeros(k)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            out[c] = curv[idx].mean()
    return out


def cluster_entropy(P, labels, k):
    out = np.zeros(k)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        H = 0.0
        for i in idx:
            for p in P[i]:
                if p > 0:
                    H += -p * math.log2(p)
        out[c] = H / len(idx)
    return out


# ============================================================
# BASIN PERSISTENCE
# ============================================================

def top10(weights):
    return np.argsort(-weights)[:10]

def basin_overlap(a, b):
    sa = set(a.tolist())
    sb = set(b.tolist())
    inter = len(sa & sb)
    return inter, inter / 10.0


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_manifold(name, baseline_seq, tunneling_seq):
    base_enc, _ = encode_states(baseline_seq)
    tun_enc,  _ = encode_states(tunneling_seq)

    n = max(base_enc.max(), tun_enc.max()) + 1

    P_base = build_markov(base_enc, n)
    P_tun  = build_markov(tun_enc, n)

    pi_base = stationary_distribution(P_base)
    pi_tun  = stationary_distribution(P_tun)

    curv_base = curvature(pi_base)
    curv_tun  = curvature(pi_tun)

    stat_drift = drift(pi_base, pi_tun)
    curv_drift = drift(curv_base, curv_tun)
    te_drift   = abs(conditional_entropy(P_base) - conditional_entropy(P_tun))

    labels_base, _ = kmeans(pi_base, N_CLUSTERS)
    labels_tun,  _ = kmeans(pi_tun,  N_CLUSTERS)

    w_base = cluster_weights(pi_base, labels_base, N_CLUSTERS)
    w_tun  = cluster_weights(pi_tun,  labels_tun,  N_CLUSTERS)

    c_base = cluster_curvature(curv_base, labels_base, N_CLUSTERS)
    c_tun  = cluster_curvature(curv_tun,  labels_tun,  N_CLUSTERS)

    H_base = cluster_entropy(P_base, labels_base, N_CLUSTERS)
    H_tun  = cluster_entropy(P_tun,  labels_tun,  N_CLUSTERS)

    top_base = top10(w_base)
    top_tun  = top10(w_tun)

    overlap_count, overlap_frac = basin_overlap(top_base, top_tun)

    return {
        "stat_drift": stat_drift,
        "curv_drift": curv_drift,
        "te_drift": te_drift,
        "top_base": top_base,
        "top_tun": top_tun,
        "overlap_count": overlap_count,
        "overlap_frac": overlap_frac,
        "w_base": w_base,
        "w_tun": w_tun,
        "c_base": c_base,
        "c_tun": c_tun,
        "H_base": H_base,
        "H_tun": H_tun,
    }


# ============================================================
# RUN
# ============================================================

def main():
    hex_base = load_hex_dataset(HEX_BASELINE_FILE)
    hex_tun  = load_hex_dataset(HEX_TUNNEL_FILE)
    byte_base = load_byte_dataset(BYTE_BASELINE_FILE)
    byte_tun  = load_byte_dataset(BYTE_TUNNEL_FILE)

    print(f"Loaded hex baseline: {len(hex_base)}")
    print(f"Loaded hex tunneling: {len(hex_tun)}")
    print(f"Loaded byte baseline: {len(byte_base)}")
    print(f"Loaded byte tunneling: {len(byte_tun)}\n")

    hex_stats  = analyze_manifold("hex", hex_base, hex_tun)
    byte_stats = analyze_manifold("byte", byte_base, byte_tun)

    print("=== Manifold-independent drift and curvature ===")
    print(f"hexagram: stationary_drift={hex_stats['stat_drift']:.6f}  "
          f"curvature_drift={hex_stats['curv_drift']:.6f}  "
          f"trans_entropy_drift={hex_stats['te_drift']:.6f}")
    print(f"byte:     stationary_drift={byte_stats['stat_drift']:.6f} "
          f"curvature_drift={byte_stats['curv_drift']:.6f} "
          f"trans_entropy_drift={byte_stats['te_drift']:.6f}\n")

    print("=== Basin persistence (top-10 clusters) ===")
    print("hexagram manifold:")
    print("  baseline top-10 clusters:", hex_stats["top_base"])
    print("  tunneling top-10 clusters:", hex_stats["top_tun"])
    print(f"  overlap_count={hex_stats['overlap_count']}  "
          f"overlap_fraction={hex_stats['overlap_frac']:.3f}")
    print("byte manifold:")
    print("  baseline top-10 clusters:", byte_stats["top_base"])
    print("  tunneling top-10 clusters:", byte_stats["top_tun"])
    print(f"  overlap_count={byte_stats['overlap_count']}  "
          f"overlap_fraction={byte_stats['overlap_frac']:.3f}\n")

    print("=== Cluster-level summaries (byte manifold, first 10 clusters) ===")
    for cid in range(10):
        print(f"cluster {cid}: weight_base={byte_stats['w_base'][cid]:.4f} "
              f"weight_tun={byte_stats['w_tun'][cid]:.4f} "
              f"curv_base={byte_stats['c_base'][cid]:.4f} "
              f"curv_tun={byte_stats['c_tun'][cid]:.4f} "
              f"H_base={byte_stats['H_base'][cid]:.4f} "
              f"H_tun={byte_stats['H_tun'][cid]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
