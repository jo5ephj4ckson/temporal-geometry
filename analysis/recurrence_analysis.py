import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================
# LOAD AND CLEAN DATA
# ============================
df = pd.read_csv("hexagram_dataset_with_rng.csv")
df = df.dropna(subset=["hexagram_lines"])

def expand_hexagram_lines(df):
    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]
    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")
    parts = parts.dropna()
    df = df.loc[parts.index]
    return pd.concat([df, parts.astype(int)], axis=1)

df = expand_hexagram_lines(df)

cols = ["line1","line2","line3","line4","line5","line6"]
X = df[cols].values

# ============================
# CLUSTER INTO BASINS
# ============================
k = 20
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

uniq = sorted(np.unique(labels))
mapping = {u:i for i,u in enumerate(uniq)}
seq = np.array([mapping[l] for l in labels])
n = len(uniq)
total = len(seq)

# ============================
# RECURRENCE METRICS
# ============================
visit_counts = np.zeros(n, dtype=int)
return_counts = np.zeros(n, dtype=int)
sum_intervals = np.zeros(n, dtype=float)

last_visit_index = [-1] * n

for t, s in enumerate(seq):
    # count visit
    visit_counts[s] += 1

    if last_visit_index[s] != -1:
        interval = t - last_visit_index[s]
        sum_intervals[s] += interval
        return_counts[s] += 1

    last_visit_index[s] = t

mean_intervals = np.zeros(n, dtype=float)
for i in range(n):
    if return_counts[i] > 0:
        mean_intervals[i] = sum_intervals[i] / return_counts[i]
    else:
        mean_intervals[i] = np.nan

self_recurrence_ratio = np.zeros(n, dtype=float)
for i in range(n):
    if visit_counts[i] > 0:
        self_recurrence_ratio[i] = return_counts[i] / visit_counts[i]
    else:
        self_recurrence_ratio[i] = np.nan

# ============================
# PRINT RECURRENCE PROFILE
# ============================
print("\n==============================")
print("RECURRENCE ANALYSIS")
print("==============================")
print(f"Total samples: {total}\n")

for i in range(n):
    print(f"Basin {uniq[i]}:")
    print(f"  Visits:              {visit_counts[i]}")
    print(f"  Returns:             {return_counts[i]}")
    print(f"  Mean return interval: {mean_intervals[i]:.2f}")
    print(f"  Self-recurrence ratio: {self_recurrence_ratio[i]:.4f}")
    print()

print("Done.")
