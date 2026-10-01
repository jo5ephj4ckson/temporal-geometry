import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

INPUT_FILE = "raw_temporal_entropy.txt"

def parse_line(line):
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None
    ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
    val = float(parts[1])
    return ts, val

def load_entropy_stream():
    timestamps = []
    values = []

    print(f"Loading entropy file: {INPUT_FILE}")

    with open(INPUT_FILE, "r") as f:
        for line in f:
            ts, val = parse_line(line)
            if ts is not None:
                timestamps.append(ts)
                values.append(val)

    return np.array(timestamps), np.array(values)

def compute_curvature(values):
    # Second derivative approximation
    curvature = np.gradient(np.gradient(values))
    return curvature

def detect_tunneling(values, curvature, threshold_factor=4.0):
    # Tunneling = sudden entropy collapse or spike
    std = np.std(values)
    threshold = std * threshold_factor

    events = []
    for i, v in enumerate(values):
        if abs(v) > threshold or abs(curvature[i]) > threshold:
            events.append(i)
    return events

def cluster_signatures(values, n_clusters=4):
    # Reshape for sklearn
    X = values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_

def drift_map(values):
    # Rolling mean drift
    window = 500
    drift = np.convolve(values, np.ones(window)/window, mode='same')
    return drift

def plot_all(timestamps, values, curvature, tunneling_events, clusters, drift):
    t = np.arange(len(values))

    plt.figure(figsize=(14, 10))

    # Entropy over time
    plt.subplot(2, 2, 1)
    plt.plot(t, values, color='blue')
    plt.title("Entropy Over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Entropy")

    # Curvature
    plt.subplot(2, 2, 2)
    plt.plot(t, curvature, color='red')
    plt.title("Temporal Curvature")
    plt.xlabel("Sample Index")
    plt.ylabel("Curvature")

    # Tunneling events
    plt.subplot(2, 2, 3)
    plt.plot(t, values, color='gray')
    plt.scatter(tunneling_events, values[tunneling_events], color='magenta')
    plt.title("Tunneling Events")
    plt.xlabel("Sample Index")
    plt.ylabel("Entropy")

    # Drift map
    plt.subplot(2, 2, 4)
    plt.plot(t, drift, color='green')
    plt.title("Temporal Drift Map")
    plt.xlabel("Sample Index")
    plt.ylabel("Drift")

    plt.tight_layout()
    plt.show()

def main():
    timestamps, values = load_entropy_stream()

    print(f"Loaded {len(values)} entropy samples.")

    curvature = compute_curvature(values)
    tunneling_events = detect_tunneling(values, curvature)
    clusters, centers = cluster_signatures(values)
    drift = drift_map(values)

    print("\n=== TEMPORAL GEOMETRY SUMMARY ===")
    print(f"Total samples: {len(values)}")
    print(f"Tunneling events detected: {len(tunneling_events)}")
    print(f"Cluster centers: {centers.flatten()}")

    plot_all(timestamps, values, curvature, tunneling_events, clusters, drift)

if __name__ == "__main__":
    main()
