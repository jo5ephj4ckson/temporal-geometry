#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

N_STATES = 32
N_CLUSTERS = 3
WINDOW_SIZES = [50, 100, 200]
DRIFT_SEGMENTS = 4


def load_df(path):
    df = pd.read_csv(path)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def build_states(series, n_states):
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


def markov_curve(states, n_states, window_size):
    feats = []
    idxs = []
    for i in range(0, len(states) - window_size):
        w = states[i:i + window_size]
        M = markov_matrix(w, n_states)
        feats.append(M.flatten())
        idxs.append(i)
    return np.array(feats), np.array(idxs)


def cluster_features(features, n_clusters):
    if len(features) == 0:
        return np.array([])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    return labels


def drift(series, time, segments):
    n = len(series)
    seg_size = n // segments
    rows = []
    for s in range(segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < segments - 1 else n
        seg = series.iloc[start:end]
        rows.append({
            "segment": s,
            "start_time": float(time.iloc[start]),
            "end_time": float(time.iloc[end - 1]),
            "mean": float(seg.mean()),
            "variance": float(seg.var()),
            "min": float(seg.min()),
            "max": float(seg.max())
        })
    return pd.DataFrame(rows)


def summarize_clusters(labels):
    if len(labels) == 0:
        return {}
    u, c = np.unique(labels, return_counts=True)
    return dict(zip(u, c))


def main():
    df = load_df(CSV_FILE)
    series = df[VALUE_COL]

    print(f"Loaded {len(df)} rows from {CSV_FILE}")

    states, bins = build_states(series, N_STATES)
    print(f"Built {N_STATES} Markov states from '{VALUE_COL}'")
    print("State bin edges:", bins)

    drift_df = drift(series, df[TIME_COL], DRIFT_SEGMENTS)
    print("\nDrift stats by segment:")
    print(drift_df)

    for w in WINDOW_SIZES:
        print(f"\n=== WINDOW SIZE {w} ===")
        feats, idxs = markov_curve(states, N_STATES, w)
        print(f"Computed {len(feats)} Markov feature windows (window_size={w})")

        labels = cluster_features(feats, N_CLUSTERS)
        if len(labels) == 0:
            print("Not enough data for clustering at this window size.")
            continue

        cluster_df = pd.DataFrame({
            "window_start_index": idxs,
            "cluster_label": labels
        })
        print(cluster_df.head())

        dist = summarize_clusters(labels)
        print("Cluster distribution:", dist)


if __name__ == "__main__":
    main()
