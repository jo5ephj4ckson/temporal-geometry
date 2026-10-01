#!/usr/bin/env python3
import csv
from collections import Counter
import os

# --- CONFIG: set your actual filenames here ---
HEX_BASELINE_FILE   = "hexagramdataset1.csv"
HEX_TUNNEL_FILE     = "tunneling_hexagram_dataset.csv"
BYTE_BASELINE_FILE  = "byte_manifold_dataset.csv"
BYTE_TUNNEL_FILE    = "tunneling_byte_manifold_dataset.csv"

# ------------------------------------------------------------
#  CHANGE COUNT EXTRACTORS
# ------------------------------------------------------------

def load_hexagram_change_counts(path):
    """Extract changing-line counts from hexagram CSV."""
    counts = Counter()
    if not os.path.exists(path):
        print(f"[WARN] Missing hexagram file: {path}")
        return counts

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "changing_lines" not in reader.fieldnames:
            print(f"[WARN] File is not a hexagram dataset: {path}")
            return counts

        for row in reader:
            cl = row["changing_lines"].strip()
            if cl.lower() == "none" or cl == "":
                k = 0
            else:
                k = len(cl.split("-"))
            counts[k] += 1

    return counts


def load_byte_change_counts(path):
    """Extract bit-change counts from byte-manifold CSV."""
    counts = Counter()
    if not os.path.exists(path):
        print(f"[WARN] Missing byte-manifold file: {path}")
        return counts

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "bit_change_count" not in reader.fieldnames:
            print(f"[WARN] File is not a byte-manifold dataset: {path}")
            return counts

        for row in reader:
            try:
                k = int(row["bit_change_count"])
                counts[k] += 1
            except:
                pass

    return counts

# ------------------------------------------------------------
#  CURVATURE CALCULATION
# ------------------------------------------------------------

def curvature_from_counts(counts, max_k):
    """Compute normalized curvature C_norm."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # probabilities p_k
    p = [(counts.get(k, 0) / total) for k in range(max_k + 1)]

    # discrete second derivative
    curv_sum = 0.0
    internal = 0
    for i in range(1, max_k):
        d2 = p[i+1] - 2*p[i] + p[i-1]
        curv_sum += abs(d2)
        internal += 1

    return curv_sum / internal if internal > 0 else 0.0

# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    results = []

    # HEXAGRAM BASELINE
    hex_base = load_hexagram_change_counts(HEX_BASELINE_FILE)
    if hex_base:
        results.append(("hexagram", "baseline_rng", curvature_from_counts(hex_base, 6)))

    # HEXAGRAM TUNNELING
    hex_tunnel = load_hexagram_change_counts(HEX_TUNNEL_FILE)
    if hex_tunnel:
        results.append(("hexagram", "tunneling_rng", curvature_from_counts(hex_tunnel, 6)))

    # BYTE BASELINE
    byte_base = load_byte_change_counts(BYTE_BASELINE_FILE)
    if byte_base:
        results.append(("byte", "baseline_rng", curvature_from_counts(byte_base, 8)))

    # BYTE TUNNELING
    byte_tunnel = load_byte_change_counts(BYTE_TUNNEL_FILE)
    if byte_tunnel:
        results.append(("byte", "tunneling_rng", curvature_from_counts(byte_tunnel, 8)))

    # PRINT RESULTS
    print("\n=== Manifold-independent curvature comparison ===")
    print("{:<10} {:<14} {:>12}".format("manifold", "entropy_source", "C_norm"))
    for manifold, source, c in results:
        print("{:<10} {:<14} {:>12.6f}".format(manifold, source, c))


if __name__ == "__main__":
    main()
