import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_points(path):
    print("Loading phase-space points:", path)
    df = pd.read_csv(path)
    print("Loaded", len(df), "points.")
    return df.values

def compute_density(points, bins=20):
    """
    Compute 3D histogram density grid.
    """
    hist, edges = np.histogramdd(points, bins=bins)
    centers = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(3)]
    return hist, centers

def plot_density_2d(hist, centers, out_prefix="phase_space_density"):
    """
    Plot 2D projections of the 3D density grid.
    """
    # XY projection
    plt.figure(figsize=(8,6))
    plt.imshow(hist.sum(axis=2).T, origin="lower", cmap="inferno",
               extent=[centers[0][0], centers[0][-1],
                       centers[1][0], centers[1][-1]])
    plt.colorbar(label="Density")
    plt.xlabel("dim_0")
    plt.ylabel("dim_1")
    plt.title("Phase-Space Density Map (XY Projection)")
    plt.savefig(f"{out_prefix}_xy.png", dpi=300)
    plt.close()
    print("Saved:", f"{out_prefix}_xy.png")

    # XZ projection
    plt.figure(figsize=(8,6))
    plt.imshow(hist.sum(axis=1).T, origin="lower", cmap="inferno",
               extent=[centers[0][0], centers[0][-1],
                       centers[2][0], centers[2][-1]])
    plt.colorbar(label="Density")
    plt.xlabel("dim_0")
    plt.ylabel("dim_2")
    plt.title("Phase-Space Density Map (XZ Projection)")
    plt.savefig(f"{out_prefix}_xz.png", dpi=300)
    plt.close()
    print("Saved:", f"{out_prefix}_xz.png")

    # YZ projection
    plt.figure(figsize=(8,6))
    plt.imshow(hist.sum(axis=0).T, origin="lower", cmap="inferno",
               extent=[centers[1][0], centers[1][-1],
                       centers[2][0], centers[2][-1]])
    plt.colorbar(label="Density")
    plt.xlabel("dim_1")
    plt.ylabel("dim_2")
    plt.title("Phase-Space Density Map (YZ Projection)")
    plt.savefig(f"{out_prefix}_yz.png", dpi=300)
    plt.close()
    print("Saved:", f"{out_prefix}_yz.png")

def plot_density_3d(hist, centers, out_file="phase_space_density_3d.png"):
    """
    Render a 3D scatter of density peaks.
    """
    x, y, z = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    # Flatten arrays
    xs, ys, zs, dens = x.flatten(), y.flatten(), z.flatten(), hist.flatten()
    mask = dens > np.percentile(dens, 95)  # top 5% density
    ax.scatter(xs[mask], ys[mask], zs[mask],
               c=dens[mask], cmap="inferno", s=10, alpha=0.6)

    ax.set_xlabel("dim_0")
    ax.set_ylabel("dim_1")
    ax.set_zlabel("dim_2")
    ax.set_title("Phase-Space Density Peaks (Top 5%)")

    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved:", out_file)

def main():
    points = load_points("temporal_phase_space_points.csv")
    hist, centers = compute_density(points, bins=20)

    print("Generating 2D density projections...")
    plot_density_2d(hist, centers)

    print("Generating 3D density peak visualization...")
    plot_density_3d(hist, centers)

    print("Done.")

if __name__ == "__main__":
    main()
