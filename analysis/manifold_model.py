#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

N_MICRO_STATES = 32
N_REGIMES = 3
WINDOW_SIZE = 100  # mid-scale resolution for manifold extraction


def load_df(path):
    df = pd.read_csv(path)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def build_micro_states(series, n_states):
    quantiles = np.linspace(0, 1, n_states + 1)
    bins = np.unique(series.quantile(quantiles).values)
    if len(bins) <= 2:
        bins = np.linspace(series.min(), series.max(), n_states + 1)
    states = np.digitize(series.values, bins[1:-1])
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
    labels = km.fit_predict(features)
    return labels


def regime_transition_matrix(labels, n_regimes):
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        T[a, b] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def regime_stats(labels, idxs, series, time, window_size, n_regimes):
    rows = []
    for r in range(n_regimes):
        mask = labels == r
        if not np.any(mask):
            continue
        starts = idxs[mask]
        means = []
        vars_ = []
        mins = []
        maxs = []
        for s in starts:
            seg = series.iloc[s:s + window_size]
            means.append(seg.mean())
            vars_.append(seg.var())
            mins.append(seg.min())
            maxs.append(seg.max())
        rows.append({
            "regime": r,
            "count": int(mask.sum()),
            "mean_mean": float(np.mean(means)),
            "mean_variance": float(np.mean(vars_)),
            "mean_min": float(np.mean(mins)),
            "mean_max": float(np.mean(maxs)),
        })
    return pd.DataFrame(rows)


def main():
    df = load_df(CSV_FILE)
    series = df[VALUE_COL]
    time = df[TIME_COL]

    print(f"Loaded {len(df)} rows from {CSV_FILE}")

    micro_states, bins = build_micro_states(series, N_MICRO_STATES)
    print(f"Built {N_MICRO_STATES} micro-states from '{VALUE_COL}'")
    print("Micro-state bin edges:", bins)

    feats, idxs = markov_feature_windows(micro_states, N_MICRO_STATES, WINDOW_SIZE)
    print(f"Computed {len(feats)} Markov feature windows (window_size={WINDOW_SIZE})")

    regimes = cluster_regimes(feats, N_REGIMES)
    print("First 10 regime labels:", regimes[:10])

    # Temporal manifold: 3-regime Markov chain
    T = regime_transition_matrix(regimes, N_REGIMES)
    print("\nTemporal manifold transition matrix (regime-level):")
    print(pd.DataFrame(T, columns=[f"to_{i}" for i in range(N_REGIMES)],
                          index=[f"from_{i}" for i in range(N_REGIMES)]))

    stats_df = regime_stats(regimes, idxs, series, time, WINDOW_SIZE, N_REGIMES)
    print("\nRegime statistics (manifold geometry):")
    print(stats_df)


if __name__ == "__main__":
    main()
