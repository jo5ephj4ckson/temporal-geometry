import csv
import math
from collections import defaultdict, Counter

DATA_FILE = "16bit_tunneling_dataset.csv"

# -----------------------------
# 1. Load dataset
# -----------------------------
timestamps = []
symbols = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = float(row["timestamp_unix"])
        s = int(row["symbol_current"])
        timestamps.append(ts)
        symbols.append(s)

n = len(symbols)
if n < 3:
    raise RuntimeError("Not enough samples for drift/curvature analysis.")

# -----------------------------
# 2. Build Markov transition counts
# -----------------------------
transition_counts = defaultdict(Counter)
state_counts = Counter()

for i in range(1, n):
    prev_s = symbols[i-1]
    cur_s = symbols[i]
    transition_counts[prev_s][cur_s] += 1
    state_counts[prev_s] += 1

# -----------------------------
# 3. Normalize to transition probabilities
# -----------------------------
transition_probs = {}
for s, targets in transition_counts.items():
    total = sum(targets.values())
    if total == 0:
        continue
    transition_probs[s] = {t: c / total for t, c in targets.items()}

# -----------------------------
# 4. Stationary distribution (empirical)
# -----------------------------
total_states = sum(state_counts.values())
stationary = {s: c / total_states for s, c in state_counts.items()}

# -----------------------------
# 5. Drift curve over time
#    Drift(t) = average |symbol(t) - symbol_baseline|
#    baseline = mean symbol over entire run
# -----------------------------
mean_symbol = sum(symbols) / n
drift_curve = []
for i, s in enumerate(symbols):
    drift = abs(s - mean_symbol)
    drift_curve.append(drift)

# -----------------------------
# 6. Curvature map
#    Curvature(t) ~ second finite difference of drift
#    k(t) = drift(t+1) - 2*drift(t) + drift(t-1)
# -----------------------------
curvature = [0.0] * n
for i in range(1, n-1):
    curvature[i] = drift_curve[i+1] - 2*drift_curve[i] + drift_curve[i-1]

# -----------------------------
# 7. Basin detection
#    Basins = top K states by stationary probability
# -----------------------------
K = 20  # number of basins to track
top_states = [s for s, _ in sorted(stationary.items(), key=lambda x: x[1], reverse=True)[:K]]

# Map each symbol to basin index or -1 if not in top K
state_to_basin = {s: i for i, s in enumerate(top_states)}
basin_sequence = []
for s in symbols:
    basin_sequence.append(state_to_basin.get(s, -1))

# -----------------------------
# 8. Basin transition matrix
#    Only between detected basins (0..K-1), ignore -1
# -----------------------------
basin_transitions = [[0 for _ in range(K)] for _ in range(K)]
basin_counts = [0 for _ in range(K)]

for i in range(1, n):
    b_prev = basin_sequence[i-1]
    b_cur = basin_sequence[i]
    if b_prev == -1 or b_cur == -1:
        continue
    basin_transitions[b_prev][b_cur] += 1
    basin_counts[b_prev] += 1

# Normalize basin transitions to probabilities
basin_transition_probs = [[0.0 for _ in range(K)] for _ in range(K)]
for i in range(K):
    total = basin_counts[i]
    if total == 0:
        continue
    for j in range(K):
        basin_transition_probs[i][j] = basin_transitions[i][j] / total

# -----------------------------
# 9. Simple text outputs
# -----------------------------
print("Samples:", n)
print("Mean symbol:", mean_symbol)
print("Drift (first 10):", drift_curve[:10])
print("Curvature (first 10):", curvature[:10])

print("\nTop basin states (by stationary probability):")
for i, s in enumerate(top_states):
    print(f"  Basin {i}: state {s}, p={stationary[s]:.6f}")

print("\nBasin transition probabilities (first 5x5):")
for i in range(min(K, 5)):
    row = basin_transition_probs[i][:min(K, 5)]
    print(f"Basin {i} ->", ["{:.3f}".format(x) for x in row])
