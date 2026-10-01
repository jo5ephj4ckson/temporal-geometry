#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

N_MICRO_STATES = 32
N_REGIMES = 3
WINDOW_SIZE = 100
TRAIN_FRACTION = 0.7  # 70% train, 30% test


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def build_micro_states(series, n_states):
    quantiles = np.linspace(0, 1, n_states + 1)
    bins = np.unique(np.quantile(series, quantiles))
    if len(bins) <= 2:
        bins = np.linspace(series.min(), series.max(), n_states + 1)
    states = np.digitize(series, bins[1:-1])
    return states, bins


def markov_matrix(states, n_states):
    M = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        s, t = states[i], states[i + 1]
        if 0 <= s < n_states and 0 <= t < n_states:
            M[s, t] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


def markov_feature_windows(states, n_states, window_size):
    feats = []
    idxs = []
    for i in range(0, len(states) - window_size):
        w = states[i:i + window_size]
        M = markov_matrix(w, n_states)
        feats.append(M.flatten())
        idxs.append(i)
    return np.array(feats), np.array(idxs)


def cluster_regimes(features, n_regimes):
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    return km.fit_predict(features)


def regime_transition_matrix(labels, n_regimes):
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def regime_stats(labels, idxs, series, window_size, n_regimes):
    stats = {}
    for r in range(n_regimes):
        mask = labels == r
        if not np.any(mask):
            continue
        starts = idxs[mask]
        means = []
        vars_ = []
        for s in starts:
            seg = series[s:s + window_size]
            means.append(seg.mean())
            vars_.append(seg.var())
        stats[r] = {"mean": float(np.mean(means)),
                    "var": float(np.mean(vars_))}
    return stats


def simulate_regimes(n_steps, T):
    regimes = np.zeros(n_steps, dtype=int)
    for i in range(1, n_steps):
        regimes[i] = np.random.choice([0, 1, 2], p=T[regimes[i - 1]])
    return regimes


def simulate_signal(regimes, stats):
    synthetic = np.zeros(len(regimes))
    for i, r in enumerate(regimes):
        synthetic[i] = np.random.normal(stats[r]["mean"], np.sqrt(stats[r]["var"]))
    return synthetic


def kl_divergence(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


def cross_entropy(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return -np.sum(p * np.log(q))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    df = pd.read_csv(CSV_FILE).sort_values(TIME_COL).reset_index(drop=True)
    series = df[VALUE_COL].values

    n = len(series)
    split = int(TRAIN_FRACTION * n)

    train_series = series[:split]
    test_series = series[split:]

    print(f"Total samples: {n}, train: {split}, test: {n - split}")

    # Build manifold on train segment
    micro_states, bins = build_micro_states(train_series, N_MICRO_STATES)
    feats, idxs = markov_feature_windows(micro_states, N_MICRO_STATES, WINDOW_SIZE)
    regimes = cluster_regimes(feats, N_REGIMES)
    T = regime_transition_matrix(regimes, N_REGIMES)
    stats = regime_stats(regimes, idxs, train_series, WINDOW_SIZE, N_REGIMES)

    print("\nRegime transition matrix (train):")
    print(T)

    print("\nRegime stats (train):")
    for r, s in stats.items():
        print(f"Regime {r}: mean={s['mean']:.6e}, var={s['var']:.6e}")

    # Simulate on test length
    test_len = len(test_series)
    sim_regimes = simulate_regimes(test_len, T)
    synthetic_series = simulate_signal(sim_regimes, stats)

    # Regime distribution metrics
    real_reg_counts = np.bincount(regimes, minlength=N_REGIMES)
    real_reg_probs = real_reg_counts / real_reg_counts.sum()

    sim_reg_counts = np.bincount(sim_regimes, minlength=N_REGIMES)
    sim_reg_probs = sim_reg_counts / sim_reg_counts.sum()

    print("\n=== REGIME DISTRIBUTION METRICS ===")
    print("Real regime probs:", real_reg_probs)
    print("Sim regime probs: ", sim_reg_probs)
    print("KL divergence:    %.6e" % kl_divergence(real_reg_probs, sim_reg_probs))
    print("Cross-entropy:    %.6e" % cross_entropy(real_reg_probs, sim_reg_probs))

    # Value distribution metrics
    hist_bins = 50
    real_hist, edges = np.histogram(test_series, bins=hist_bins, density=True)
    sim_hist, _ = np.histogram(synthetic_series, bins=edges, density=True)

    print("\n=== VALUE DISTRIBUTION METRICS ===")
    print("KL divergence:    %.6e" % kl_divergence(real_hist, sim_hist))
    print("Cross-entropy:    %.6e" % cross_entropy(real_hist, sim_hist))
    print("Mean Squared Err: %.6e" % mean_squared_error(test_series, synthetic_series))

    # Drift / variance
    print("\n=== DRIFT / VARIANCE (TEST SEGMENT) ===")
    print("Real drift:      %.6e" % np.mean(test_series))
    print("Synthetic drift: %.6e" % np.mean(synthetic_series))
    print("Real variance:   %.6e" % np.var(test_series))
    print("Synth variance:  %.6e" % np.var(synthetic_series))


if __name__ == "__main__":
    main()
