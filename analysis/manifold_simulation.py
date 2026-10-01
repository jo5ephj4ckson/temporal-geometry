#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

# ---------------------------------------------------------
# 1. Load real data
# ---------------------------------------------------------
df = pd.read_csv(CSV_FILE)
df = df.sort_values(TIME_COL).reset_index(drop=True)
real_series = df[VALUE_COL].values

print(f"Loaded {len(real_series)} real samples.")


# ---------------------------------------------------------
# 2. Temporal manifold parameters (from your output)
# ---------------------------------------------------------
# Regime transition matrix (3x3)
T = np.array([
    [0.997365, 0.002635, 0.000000],
    [0.001580, 0.996840, 0.001580],
    [0.002000, 0.000000, 0.998000]
])

# Regime geometry (mean drift, variance, min, max)
regime_stats = {
    0: {"mean": -0.000011, "var": 7.232916e-10, "min": -0.000065, "max": 0.000043},
    1: {"mean": -0.000013, "var": 6.644435e-10, "min": -0.000066, "max": 0.000040},
    2: {"mean": -0.000013, "var": 7.323374e-10, "min": -0.000067, "max": 0.000041},
}

print("Loaded temporal manifold parameters.")


# ---------------------------------------------------------
# 3. Simulate regime transitions
# ---------------------------------------------------------
def simulate_regimes(n_steps, T):
    regimes = np.zeros(n_steps, dtype=int)
    for i in range(1, n_steps):
        regimes[i] = np.random.choice([0, 1, 2], p=T[regimes[i-1]])
    return regimes


# ---------------------------------------------------------
# 4. Generate synthetic temporal signal
# ---------------------------------------------------------
def simulate_signal(regimes, stats):
    synthetic = np.zeros(len(regimes))
    for i, r in enumerate(regimes):
        mean = stats[r]["mean"]
        var = stats[r]["var"]
        synthetic[i] = np.random.normal(mean, np.sqrt(var))
    return synthetic


# ---------------------------------------------------------
# 5. Run simulation
# ---------------------------------------------------------
n_steps = len(real_series)
regimes = simulate_regimes(n_steps, T)
synthetic_series = simulate_signal(regimes, regime_stats)

print("Simulation complete.")


# ---------------------------------------------------------
# 6. Compare synthetic vs real
# ---------------------------------------------------------
mse = mean_squared_error(real_series, synthetic_series)
corr = np.corrcoef(real_series, synthetic_series)[0, 1]

print("\n=== MODEL vs REAL DATA ===")
print(f"Mean Squared Error: {mse:.6e}")
print(f"Correlation:        {corr:.4f}")

# Drift comparison
real_drift = np.mean(real_series)
synthetic_drift = np.mean(synthetic_series)

print("\nDrift comparison:")
print(f"Real drift:      {real_drift:.6e}")
print(f"Synthetic drift: {synthetic_drift:.6e}")

# Variance comparison
real_var = np.var(real_series)
synthetic_var = np.var(synthetic_series)

print("\nVariance comparison:")
print(f"Real variance:      {real_var:.6e}")
print(f"Synthetic variance: {synthetic_var:.6e}")

# Min/max comparison
print("\nMin/Max comparison:")
print(f"Real min/max:      {real_series.min():.6e}, {real_series.max():.6e}")
print(f"Synthetic min/max: {synthetic_series.min():.6e}, {synthetic_series.max():.6e}")
