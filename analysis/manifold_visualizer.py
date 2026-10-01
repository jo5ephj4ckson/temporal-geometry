#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

# ---------------------------------------------------------
# 1. Load real data
# ---------------------------------------------------------
df = pd.read_csv(CSV_FILE)
df = df.sort_values(TIME_COL).reset_index(drop=True)
real_series = df[VALUE_COL].values
time = df[TIME_COL].values

print(f"Loaded {len(real_series)} real samples.")


# ---------------------------------------------------------
# 2. Temporal manifold parameters (from your model)
# ---------------------------------------------------------
T = np.array([
    [0.997365, 0.002635, 0.000000],
    [0.001580, 0.996840, 0.001580],
    [0.002000, 0.000000, 0.998000]
])

regime_stats = {
    0: {"mean": -0.000011, "var": 7.232916e-10},
    1: {"mean": -0.000013, "var": 6.644435e-10},
    2: {"mean": -0.000013, "var": 7.323374e-10},
}

# ---------------------------------------------------------
# 3. Simulate regimes
# ---------------------------------------------------------
def simulate_regimes(n_steps, T):
    regimes = np.zeros(n_steps, dtype=int)
    for i in range(1, n_steps):
        regimes[i] = np.random.choice([0, 1, 2], p=T[regimes[i-1]])
    return regimes

# ---------------------------------------------------------
# 4. Generate synthetic signal
# ---------------------------------------------------------
def simulate_signal(regimes, stats):
    synthetic = np.zeros(len(regimes))
    for i, r in enumerate(regimes):
        synthetic[i] = np.random.normal(stats[r]["mean"], np.sqrt(stats[r]["var"]))
    return synthetic

regimes = simulate_regimes(len(real_series), T)
synthetic_series = simulate_signal(regimes, regime_stats)

print("Simulation complete.")


# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------

plt.figure(figsize=(14, 4))
plt.plot(time, regimes, lw=0.5)
plt.title("Temporal Manifold Regime Timeline")
plt.xlabel("Time")
plt.ylabel("Regime (0,1,2)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Drift curves
plt.figure(figsize=(14, 4))
plt.plot(time, real_series, label="Real", alpha=0.7)
plt.plot(time, synthetic_series, label="Synthetic", alpha=0.7)
plt.title("Real vs Synthetic Drift Curve")
plt.xlabel("Time")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Regime occupancy
unique, counts = np.unique(regimes, return_counts=True)
plt.figure(figsize=(6, 4))
sns.barplot(x=unique, y=counts)
plt.title("Regime Occupancy (Manifold Structure)")
plt.xlabel("Regime")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Transition matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(T, annot=True, cmap="Blues", fmt=".4f")
plt.title("Temporal Manifold Transition Matrix")
plt.xlabel("To Regime")
plt.ylabel("From Regime")
plt.tight_layout()
plt.show()
