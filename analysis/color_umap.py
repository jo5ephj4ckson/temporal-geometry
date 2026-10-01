import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
df = pd.read_csv("structured_output.csv")

# === Select features ===
features = df[[
    'line1','line2','line3','line4','line5','line6',
    'cpu_percent','ram_percent','disk_percent',
    'time_since_process_start_seconds','system_boot_time_unix'
]]

# === Scale features ===
scaler = StandardScaler()
X = scaler.fit_transform(features)

# === Run HDBSCAN ===
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=500,
    min_samples=50,
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(X)
df['hdbscan_cluster'] = labels

# === Run UMAP ===
reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
    n_components=2,
    metric='euclidean'
)
embedding = reducer.fit_transform(X)
df['umap_x'] = embedding[:,0]
df['umap_y'] = embedding[:,1]

# === Plot UMAP colored by HDBSCAN cluster ===
plt.figure(figsize=(10,7))
scatter = plt.scatter(df['umap_x'], df['umap_y'], c=df['hdbscan_cluster'], s=1, cmap='Spectral')
plt.title("UMAP Projection Colored by HDBSCAN Cluster")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.colorbar(scatter, label='HDBSCAN Cluster ID')
plt.tight_layout()
plt.show()
