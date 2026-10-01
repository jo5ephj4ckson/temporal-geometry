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

# === Try multiple k values ===
ks = [4, 6, 8, 12, 16]
inertias = []

for k in ks:
    print(f"Running k-means for k={k}...")
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# === Plot elbow curve ===
plt.plot(ks, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# === Choose a k value (example: 8) ===
k = 8
km = KMeans(n_clusters=k, random_state=42)
df['cluster'] = km.fit_predict(X)

# === Save clustered dataset ===
df.to_csv("clustered_output.csv", index=False)
print("Clustered dataset saved as clustered_output.csv")

# === Plot temporal drift ===
plt.figure(figsize=(12,6))
plt.plot(df['timestamp_unix'], df['cluster'], '.', markersize=1)
plt.xlabel("Unix Time")
plt.ylabel("Cluster")
plt.title("Temporal Drift Across Clusters")
plt.show()
