import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------- CONFIG --------
CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"  # you can switch to 'variance' or others
N_STATES = 8        # number of Markov states (bins)
N_CLUSTERS = 3      # number of clusters for Markov feature vectors
DRIFT_WINDOWS = 4   # number of segments for drift analysis
# ------------------------


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # sort by time just in case
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def build_markov_states(series, n_states):
    # bin continuous values into discrete states
    # using quantiles for roughly balanced bins
    quantiles = np.linspace(0, 1, n_states + 1)
    bins = series.quantile(quantiles).values
    # ensure uniqueness
    bins = np.unique(bins)
    # if quantiles collapse, fall back to linspace
    if len(bins) <= 2:
        bins = np.linspace(series.min(), series.max(), n_states + 1)
    state_ids = np.digitize(series.values, bins[1:-1], right=False)
    return state_ids, bins


def compute_markov_transition_matrix(states, n_states):
    # states are integers in [0, n_states-1]
    trans = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        s = states[i]
        t = states[i + 1]
        if 0 <= s < n_states and 0 <= t < n_states:
            trans[s, t] += 1.0
    # normalize rows to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans /= row_sums
    return trans


def build_markov_feature_curve(states, n_states, window_size):
    """
    Slide over the state sequence and compute transition matrices
    per window; flatten each matrix into a feature vector.
    """
    features = []
    indices = []
    for start in range(0, len(states) - window_size):
        end = start + window_size
        window_states = states[start:end]
        trans = compute_markov_transition_matrix(window_states, n_states)
        feat = trans.flatten()
        features.append(feat)
        indices.append(start)
    features = np.array(features)
    return features, np.array(indices)


def cluster_markov_features(features, n_clusters):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km, scaler


def drift_analysis(series, time_values, n_segments):
    """
    Simple drift: split time into segments and compute mean/variance per segment.
    """
    n = len(series)
    seg_size = n // n_segments
    stats = []
    for i in range(n_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_segments - 1 else n
        seg = series.iloc[start:end]
        t_start = time_values.iloc[start]
        t_end = time_values.iloc[end - 1]
        stats.append({
            "segment": i,
            "start_time": t_start,
            "end_time": t_end,
            "mean": float(seg.mean()),
            "variance": float(seg.var()),
            "min": float(seg.min()),
            "max": float(seg.max())
        })
    return pd.DataFrame(stats)


def plot_cluster_curve(indices, labels):
    plt.figure(figsize=(10, 4))
    plt.plot(indices, labels, marker="o", linestyle="-", alpha=0.7)
    plt.xlabel("Window start index")
    plt.ylabel("Cluster label")
    plt.title("Cluster Markov Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drift(df_stats):
    plt.figure(figsize=(10, 4))
    plt.plot(df_stats["segment"], df_stats["mean"], marker="o", label="mean")
    plt.plot(df_stats["segment"], df_stats["variance"], marker="o", label="variance")
    plt.xlabel("Segment")
    plt.ylabel("Value")
    plt.title("Drift over segments (mean & variance)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # allow optional filename override: python markov_cluster_drift.py myfile.csv
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = CSV_FILE

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # --- Markov states from VALUE_COL ---
    series = df[VALUE_COL]
    states, bins = build_markov_states(series, N_STATES)
    print(f"Built {N_STATES} Markov states from '{VALUE_COL}'")
    print("State bin edges:", bins)

    # --- Markov feature curve ---
    # window size: small fraction of length, but at least 20
    window_size = max(20, len(states) // 50)
    features, indices = build_markov_feature_curve(states, N_STATES, window_size)
    print(f"Computed {len(features)} Markov feature windows (window_size={window_size})")

    # --- Clustering ---
    labels, km, scaler = cluster_markov_features(features, N_CLUSTERS)
    print(f"Clustered into {N_CLUSTERS} clusters")
    # attach to dataframe (approximate alignment)
    df_markov = pd.DataFrame({
        "window_start_index": indices,
        "cluster_label": labels
    })
    print(df_markov.head())

    # --- Drift analysis ---
    drift_df = drift_analysis(series, df[TIME_COL], DRIFT_WINDOWS)
    print("\nDrift stats by segment:")
    print(drift_df)

    # --- Plots ---
    plot_cluster_curve(indices, labels)
    plot_drift(drift_df)


if __name__ == "__main__":
    main()
