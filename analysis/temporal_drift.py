import pandas as pd
import matplotlib.pyplot as plt

# === Load clustered dataset ===
df = pd.read_csv("clustered_output.csv")

# === Optional: sort by time just to be safe ===
df = df.sort_values("timestamp_unix")

# === Plot temporal drift ===
plt.figure(figsize=(14, 6))

plt.scatter(
    df['timestamp_unix'],
    df['cluster'],
    s=4,
    c=df['cluster'],
    cmap='tab10'
)

plt.xlabel("Unix Time")
plt.ylabel("Cluster (Temporal State)")
plt.title("Temporal Drift Across Probe Runtime (k=8)")
plt.tight_layout()
plt.show()
