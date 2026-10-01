import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_points(path):
    print("Loading phase-space points:", path)
    df = pd.read_csv(path)
    print("Loaded", len(df), "points.")
    return df.values

def box_counting(points, min_box_size=1, max_box_size=20, num_scales=10):
    """
    Estimate fractal dimension using box-counting method.
    """
    box_sizes = np.linspace(min_box_size, max_box_size, num_scales)
    counts = []

    for size in box_sizes:
        # Compute box indices for each point
        indices = np.floor(points / size).astype(int)
        # Count unique boxes occupied
        unique_boxes = np.unique(indices, axis=0)
        counts.append(len(unique_boxes))

    box_sizes = np.array(box_sizes)
    counts = np.array(counts)

    # Fit slope of log(counts) vs log(1/box_size)
    x = np.log(1.0 / box_sizes)
    y = np.log(counts)
    coeffs = np.polyfit(x, y, 1)
    fractal_dim = coeffs[0]

    return fractal_dim, box_sizes, counts

def plot_fractal_dimension(box_sizes, counts, fractal_dim, out_file="fractal_dimension_plot.png"):
    plt.figure(figsize=(8,6))
    plt.plot(np.log(1.0 / box_sizes), np.log(counts), 'o-', color='darkorange', alpha=0.8)
    plt.title(f"Fractal Dimension ≈ {fractal_dim:.5f}")
    plt.xlabel("log(1 / box size)")
    plt.ylabel("log(box count)")
    plt.grid(True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved:", out_file)

def main():
    points = load_points("temporal_phase_space_points.csv")
    fractal_dim, box_sizes, counts = box_counting(points, min_box_size=1, max_box_size=20, num_scales=10)
    print(f"Estimated fractal dimension: {fractal_dim:.6f}")
    plot_fractal_dimension(box_sizes, counts, fractal_dim)
    print("Done.")

if __name__ == "__main__":
    main()
