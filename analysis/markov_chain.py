import pandas as pd
import numpy as np

# Load the clustered dataset
df = pd.read_csv("structured_output_clustered.csv")

# Use the actual column name from your file
states = df["cluster"].values

unique_states = sorted(df["cluster"].unique())
n_states = len(unique_states)

# --- 1. Build transition counts ---
transition_counts = np.zeros((n_states, n_states), dtype=int)

for i in range(len(states) - 1):
    a = states[i]
    b = states[i + 1]
    transition_counts[a, b] += 1

# --- 2. Normalize into transition probabilities ---
transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

# --- 3. Compute stationary distribution ---
eigvals, eigvecs = np.linalg.eig(transition_probs.T)
idx = np.argmin(np.abs(eigvals - 1))
stationary = np.real(eigvecs[:, idx])
stationary = stationary / stationary.sum()

# --- 4. Save results ---
np.savetxt("transition_counts.csv", transition_counts, delimiter=",")
np.savetxt("transition_probs.csv", transition_probs, delimiter=",")
np.savetxt("stationary_distribution.csv", stationary, delimiter=",")

print("States:", unique_states)
print("Stationary distribution:", stationary)
