import csv
from collections import Counter
import os

HEXAGRAM_FILE = "hexagramdataset1.csv"
BYTE_FILE = "byte_manifold_dataset.csv"

def load_hexagram_changing_counts(path: str):
    """
    Load changing-line counts from hexagramdataset1.csv.
    Assumes column 'changing_lines' contains either 'none' or
    a '-' separated list of line indices (1-6).
    """
    counts = Counter()
    total = 0

    if not os.path.exists(path):
        print(f"[hex] file not found: {path}")
        return counts, total

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cl = row.get("changing_lines", "").strip()
            if cl.lower() == "none" or cl == "":
                n = 0
            else:
                n = len(cl.split("-"))
            counts[n] += 1
            total += 1

    return counts, total

def load_byte_bit_change_counts(path: str):
    """
    Load bit-change counts from byte_manifold_dataset.csv.
    Assumes column 'bit_change_count' is an integer 0-8.
    """
    counts = Counter()
    total = 0

    if not os.path.exists(path):
        print(f"[byte] file not found: {path}")
        return counts, total

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bc_str = row.get("bit_change_count", "").strip()
            if bc_str == "":
                continue
            try:
                n = int(bc_str)
            except ValueError:
                continue
            counts[n] += 1
            total += 1

    return counts, total

def print_distribution(label: str, counts: Counter, total: int, max_k: int):
    print(f"\n=== {label} ===")
    print(f"Total samples: {total}")
    for k in range(0, max_k + 1):
        c = counts.get(k, 0)
        p = (c / total) if total > 0 else 0.0
        print(f"  {k}: count={c}  prob={p:.6f}")

def main():
    # --- hexagram manifold ---
    hex_counts, hex_total = load_hexagram_changing_counts(HEXAGRAM_FILE)
    print_distribution("Hexagram manifold changing-line distribution (0–6)", hex_counts, hex_total, 6)

    # --- byte manifold ---
    byte_counts, byte_total = load_byte_bit_change_counts(BYTE_FILE)
    print_distribution("Byte manifold bit-change distribution (0–8)", byte_counts, byte_total, 8)

    # --- quick qualitative comparison hint ---
    # You will visually compare:
    # - peak location (which k has highest probability)
    # - tail behavior (how fast probabilities drop toward 0 and max)
    # - multi-modality (single peak vs multiple peaks)

if __name__ == "__main__":
    main()
