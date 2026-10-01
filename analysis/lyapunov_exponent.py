import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def load_points(path):
    print("Loading phase-space points:", path)
    df = pd.read_csv(path)
    print("Loaded", len(df), "points.")
    return df.values

def estimate_lyapunov(points, max_iter=50, neighbor_radius=0.1):
    """
    Estimate the largest Lyapunov exponent using Rosenstein's algorithm.
    """
    n_points = len(points)
    dim = points.shape[1]

    # Find nearest neighbors (excluding self)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    nearest = indices[:, 1]  # second neighbor (first is self)

    # Initialize divergence array
    divergence = np.zeros(max_iter)

    for i in range(n_points - max_iter):
        for j in range(max_iter):
            p1 = points[i + j]
            p2 = points[nearest[i] + j] if nearest[i] + j < n_points else points[nearest[i]]
            divergence[j] += np.linalg.norm(p1 - p2)

    divergence /= (n_points - max_iter)
    divergence = np.where(divergence == 0, np.nan, divergence)

    # Compute Lyapunov exponent from slope of log(divergence)
    valid = ~np.isnan(divergence)
    x = np.arange(max_iter)[valid]
    y = np.log(divergence[valid])

    coeffs = np.polyfit(x, y, 1)
    lyap_exp = coeffs[0]

    return lyap_exp, x, y

def plot_lyapunov(x, y, lyap_exp, out_file="lyapunov_exponent_plot.png"):
    plt.figure(figsize=(8,6))
    plt.plot(x, y, 'o-', color='purple', alpha=0.7)
    plt.title(f"Largest Lyapunov Exponent ≈ {lyap_exp:.5f}")
    plt.xlabel("Iteration (time steps)")
    plt.ylabel("log(divergence)")
    plt.grid(True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved:", out_file)

def main():
    points = load_points("temporal_phase_space_points.csv")
    lyap_exp, x, y = estimate_lyapunov(points, max_iter=50)
    print(f"Estimated Lyapunov exponent: {lyap_exp:.6f}")
    plot_lyapunov(x, y, lyap_exp)
    print("Done.")

if __name__ == "__main__":
    main()
