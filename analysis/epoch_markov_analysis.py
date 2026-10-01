import pandas as pd
import numpy as np

# ============================
# LOAD CLUSTERED DATA
# ============================
df = pd.read_csv("structured_output_clustered.csv")

states = df["cluster"].values
unique_states = sorted(df["cluster"].unique())
n_states = len(unique_states)

# Map cluster labels to 0..N-1 for matrix indexing
label_to_idx = {label: i for i, label in enumerate(unique_states)}
idx_states = np.array([label_to_idx[s] for s in states])

# ============================
# EPOCH SPLITTING
# ============================
N = len(idx_states)
epoch_size = N // 3

epochs = {
    "epoch_1": idx_states[:epoch_size],
    "epoch_2": idx_states[epoch_size:2*epoch_size],
    "epoch_3": idx_states[2*epoch_size:]
}

def build_markov_chain(state_seq, n_states):
    """Build transition matrix + stationary distribution."""
    T = np.zeros((n_states, n_states), dtype=float)

    for i in range(len(state_seq) - 1):
        a = state_seq[i]
        b = state_seq[i + 1]
        T[a, b] += 1

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()

    return P, stationary

# ============================
# RUN MARKOV CHAINS PER EPOCH
# ============================
results = {}

for name, seq in epochs.items():
    P, stationary = build_markov_chain(seq, n_states)
    results[name] = {
        "P": P,
        "stationary": stationary
    }

# ============================
# ATTRACTOR IDENTIFICATION
# ============================
def top_attractors(stationary, top_k=10):
    idxs = np.argsort(stationary)[::-1][:top_k]
    return [(unique_states[i], stationary[i]) for i in idxs]

# ============================
# PRINT RESULTS
# ============================
for name in results:
    print("\n==============================")
    print(f"{name.upper()} RESULTS")
    print("==============================")

    stationary = results[name]["stationary"]

    print("\nTop attractor states:")
    for label, prob in top_attractors(stationary):
        print(f"  State {label}: {prob:.6f}")

# ============================
# EPOCH DRIFT ANALYSIS
# ============================
print("\n\n==============================")
print("EPOCH DRIFT ANALYSIS")
print("==============================")

s1 = results["epoch_1"]["stationary"]
s2 = results["epoch_2"]["stationary"]
s3 = results["epoch_3"]["stationary"]

def drift(a, b):
    return np.sum(np.abs(a - b))

print(f"Drift epoch_1 → epoch_2: {drift(s1, s2):.6f}")
print(f"Drift epoch_2 → epoch_3: {drift(s2, s3):.6f}")
print(f"Drift epoch_1 → epoch_3: {drift(s1, s3):.6f}")

print("\nDone.")
