import csv
import random
from collections import defaultdict, Counter

DATA_FILE = "16bit_tunneling_dataset.csv"

# -----------------------------
# 1. Load timestamps from real dataset
# -----------------------------
timestamps = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamps.append(float(row["timestamp_unix"]))

n = len(timestamps)
print("Loaded", n, "timestamps.")

# -----------------------------
# 2. Generate synthetic RNG symbols
# -----------------------------
symbols = [random.randint(0, 65535) for _ in range(n)]

# -----------------------------
# 3. Drift curve
# -----------------------------
mean_symbol = sum(symbols) / n
drift_curve = [abs(s - mean_symbol) for s in symbols]

# -----------------------------
# 4. Curvature (second derivative)
# -----------------------------
curvature = [0.0] * n
for i in range(1, n-1):
    curvature[i] = drift_curve[i+1] - 2*drift_curve[i] + drift_curve[i-1]

# -----------------------------
# 5. Basin detection
# -----------------------------
state_counts = Counter(symbols)
total_states = sum(state_counts.values())
stationary = {s: c / total_states for s, c in state_counts.items()}

K = 20
top_states = [s for s, _ in sorted(stationary.items(), key=lambda x: x[1], reverse=True)[:K]]
state_to_basin = {s: i for i, s in enumerate(top_states)}

basin_sequence = [state_to_basin.get(s, -1) for s in symbols]

# -----------------------------
# 6. Basin transition matrix
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

basin_transition_probs = [[0.0 for _ in range(K)] for _ in range(K)]
for i in range(K):
    total = basin_counts[i]
    if total == 0:
        continue
    for j in range(K):
        basin_transition_probs[i][j] = basin_transitions[i][j] / total

# -----------------------------
# 7. Output summary
# -----------------------------
print("\nSYNTHETIC RNG DRIFT (first 10):")
print(drift_curve[:10])

print("\nSYNTHETIC RNG CURVATURE (first 10):")
print(curvature[:10])

print("\nTop basin states (synthetic RNG):")
for i, s in enumerate(top_states):
    print(f"  Basin {i}: state {s}, p={stationary[s]:.6f}")

print("\nBasin transition probabilities (first 5x5):")
for i in range(min(K, 5)):
    row = basin_transition_probs[i][:min(K, 5)]
    print(f"Basin {i} ->", ["{:.3f}".format(x) for x in row])
