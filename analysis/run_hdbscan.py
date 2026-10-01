import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load your actual dataset ===
df = pd.read_csv("structured_output.csv")

# === Use ONLY columns that exist ===
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

# === Plot cluster distribution ===
plt.figure(figsize=(10,6))
plt.hist(labels, bins=len(np.unique(labels)))
plt.title("HDBSCAN Cluster Distribution")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === Save output ===
df.to_csv("hdbscan_output.csv", index=False)
