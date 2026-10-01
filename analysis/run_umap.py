import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
df = pd.read_csv("structured_output.csv")

# === Select features that exist ===
features = df[[
    'line1','line2','line3','line4','line5','line6',
    'cpu_percent','ram_percent','disk_percent',
    'time_since_process_start_seconds','system_boot_time_unix'
]]

# === Scale features ===
scaler = StandardScaler()
X = scaler.fit_transform(features)

# === Run UMAP ===
reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
    n_components=2,
    metric='euclidean'
)

embedding = reducer.fit_transform(X)

# === Plot UMAP ===
plt.figure(figsize=(10,7))
plt.scatter(embedding[:,0], embedding[:,1], s=1, alpha=0.5)
plt.title("UMAP Projection of Temporal Probe Dataset")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()
plt.show()

# === Save embedding ===
df['umap_x'] = embedding[:,0]
df['umap_y'] = embedding[:,1]
df.to_csv("umap_output.csv", index=False)
