import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_points(path):
    print("Loading phase-space points:", path)
    df = pd.read_csv(path)
    print("Loaded", len(df), "points.")
    return df.values

def plot_2d(points, out_prefix="phase_space_2d"):
    dims = points.shape[1]

    for i in range(dims):
        for j in range(i+1, dims):
            plt.figure(figsize=(8,6))
            plt.scatter(points[:, i], points[:, j],
                        s=1, alpha=0.5, c=np.arange(len(points)),
                        cmap="viridis")
            plt.xlabel(f"dim_{i}")
            plt.ylabel(f"dim_{j}")
            plt.title(f"Phase Space Projection: dim_{i} vs dim_{j}")
            fname = f"{out_prefix}_{i}_{j}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print("Saved:", fname)

def plot_3d(points, out_file="phase_space_3d.png"):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:,0], points[:,1], points[:,2],
               s=1, alpha=0.5, c=np.arange(len(points)),
               cmap="plasma")

    ax.set_xlabel("dim_0")
    ax.set_ylabel("dim_1")
    ax.set_zlabel("dim_2")
    ax.set_title("3D Temporal Phase-Space Reconstruction")

    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved:", out_file)

def main():
    points = load_points("temporal_phase_space_points.csv")

    print("Generating 2D projections...")
    plot_2d(points)

    print("Generating 3D attractor plot...")
    plot_3d(points)

    print("Done.")

if __name__ == "__main__":
    main()
