import pandas as pd
import matplotlib.pyplot as plt

# === Load clustered dataset ===
df = pd.read_csv("clustered_output.csv")
df = df.sort_values("timestamp_unix")

# === Split into early and late epochs ===
split_point = df['timestamp_unix'].quantile(0.5)
early = df[df['timestamp_unix'] <= split_point]
late = df[df['timestamp_unix'] > split_point]

# === Compute cluster frequency ===
early_counts = early['cluster'].value_counts().sort_index()
late_counts = late['cluster'].value_counts().sort_index()

# === Combine into a DataFrame ===
comparison = pd.DataFrame({
    'Early Epoch': early_counts,
    'Late Epoch': late_counts
}).fillna(0)

# === Plot comparison ===
comparison.plot(kind='bar', figsize=(10,6))
plt.title("Cluster Frequency: Early vs Late Epochs")
plt.xlabel("Cluster (Temporal State)")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.show()
