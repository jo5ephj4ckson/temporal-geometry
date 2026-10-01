import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load clustered dataset ===
df = pd.read_csv("clustered_output.csv")

# === Select numeric columns ===
numeric_cols = [
    'line1', 'line2', 'line3', 'line4', 'line5', 'line6',
    'cpu_percent', 'ram_percent', 'disk_percent',
    'time_since_process_start_seconds',
    'system_boot_time_unix',
    'timestamp_unix'
]

# === Compute centroids ===
centroids = df.groupby('cluster')[numeric_cols].mean()

# === Normalize each column for visualization ===
centroids_norm = (centroids - centroids.min()) / (centroids.max() - centroids.min())

# === Plot normalized heatmap ===
plt.figure(figsize=(12, 8))
plt.imshow(centroids_norm.values, cmap='viridis', aspect='auto')

# Add colorbar
plt.colorbar(label='Normalized Feature Value')

# Add labels
plt.xticks(
    ticks=np.arange(len(numeric_cols)),
    labels=numeric_cols,
    rotation=45,
    ha='right'
)
plt.yticks(
    ticks=np.arange(len(centroids.index)),
    labels=[f"Cluster {i}" for i in centroids.index]
)

plt.title("Normalized K-Means Centroid Heatmap (k=8)")
plt.tight_layout()
plt.show()
