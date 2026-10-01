import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === Load your structured dataset ===
df = pd.read_csv("structured_output.csv")

# === Select numeric columns for clustering ===
numeric_cols = [
    'line1', 'line2', 'line3', 'line4', 'line5', 'line6',
    'cpu_percent', 'ram_percent', 'disk_percent',
    'time_since_process_start_seconds',
    'system_boot_time_unix',
    'timestamp_unix'
]

X = df[numeric_cols]

# === Run k-means with k = 8 ===
print("Running k-means for k=8...")
km = KMeans(n_clusters=8, random_state=42)
df['cluster'] = km.fit_predict(X)

# === Save clustered dataset ===
df.to_csv("clustered_output.csv", index=False)
print("Clustered dataset saved as clustered_output.csv")

# === Plot temporal drift ===
plt.figure(figsize=(12,6))
plt.plot(df['timestamp_unix'], df['cluster'], '.', markersize=1)
plt.xlabel("Unix Time")
plt.ylabel("Cluster")
plt.title("Temporal Drift Across Clusters (k=8)")
plt.show()

# === Print cluster centroids ===
centroids = pd.DataFrame(km.cluster_centers_, columns=numeric_cols)
print("\nCluster centroids:")
print(centroids)
