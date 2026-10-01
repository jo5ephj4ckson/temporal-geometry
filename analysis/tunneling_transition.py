import csv
import numpy as np
from collections import defaultdict

# ============================================================
# Load hexagrams from your probe CSV
# ============================================================

def load_hexagrams_from_csv(path):
    hexagrams = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hexagram = "-".join([
                row["line1"], row["line2"], row["line3"],
                row["line4"], row["line5"], row["line6"]
            ])
            hexagrams.append(hexagram)
    return hexagrams

# ============================================================
# Build transition count matrix
# ============================================================

def build_transition_counts(hexagrams):
    unique = sorted(set(hexagrams))
    index = {h: i for i, h in enumerate(unique)}
    N = len(unique)
    counts = np.zeros((N, N), dtype=np.int64)

    for t in range(len(hexagrams) - 1):
        i = index[hexagrams[t]]
        j = index[hexagrams[t + 1]]
        counts[i, j] += 1

    return unique, index, counts

# ============================================================
# Normalize counts into probabilities
# ============================================================

def normalize_counts(counts):
    probs = np.zeros_like(counts, dtype=float)
    for i in range(counts.shape[0]):
        row_sum = counts[i].sum()
        if row_sum > 0:
            probs[i] = counts[i] / row_sum
    return probs

# ============================================================
# Extract useful summaries
# ============================================================

def top_transitions(unique, probs, k=20):
    results = []
    for i in range(len(unique)):
        for j in range(len(unique)):
            p = probs[i, j]
            if p > 0:
                results.append((p, unique[i], unique[j]))
    results.sort(reverse=True)
    return results[:k]

def strongest_attractors(unique, probs, k=20):
    # Self-transition probabilities P[i,i]
    attractors = []
    for i in range(len(unique)):
        attractors.append((probs[i, i], unique[i]))
    attractors.sort(reverse=True)
    return attractors[:k]

def forbidden_transitions(unique, counts, k=20):
    # Transitions that never occur
    forbidden = []
    for i in range(len(unique)):
        for j in range(len(unique)):
            if counts[i, j] == 0:
                forbidden.append((unique[i], unique[j]))
    return forbidden[:k]

def basin_candidates(unique, probs, k=20):
    # Rows with highest total outgoing probability concentration
    basin_scores = []
    for i in range(len(unique)):
        row = probs[i]
        score = np.max(row)
        basin_scores.append((score, unique[i]))
    basin_scores.sort(reverse=True)
    return basin_scores[:k]

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    path = r"C:\Users\PSNet Cannabis\Downloads\structured_output.csv"

    hexagrams = load_hexagrams_from_csv(path)
    print(f"Loaded {len(hexagrams)} hexagrams")

    unique, index, counts = build_transition_counts(hexagrams)
    probs = normalize_counts(counts)

    print("\n=== TOP TRANSITIONS ===")
    for p, h1, h2 in top_transitions(unique, probs):
        print(f"{h1} -> {h2}: {p:.6f}")

    print("\n=== STRONGEST ATTRACTORS (self-transitions) ===")
    for p, h in strongest_attractors(unique, probs):
        print(f"{h}: {p:.6f}")

    print("\n=== SAMPLE FORBIDDEN TRANSITIONS ===")
    for h1, h2 in forbidden_transitions(unique, counts):
        print(f"{h1} -> {h2}")

    print("\n=== BASIN CANDIDATES ===")
    for score, h in basin_candidates(unique, probs):
        print(f"{h}: {score:.6f}")
