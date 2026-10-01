import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_and_expand(path):
    print("\nLoading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))

    if "hexagram_lines" not in df.columns:
        df["hexagram_lines"] = (
            df["line1"].astype(str) + "-" +
            df["line2"].astype(str) + "-" +
            df["line3"].astype(str) + "-" +
            df["line4"].astype(str) + "-" +
            df["line5"].astype(str) + "-" +
            df["line6"].astype(str)
        )

    parts = df["hexagram_lines"].str.split("-", expand=True)
    parts.columns = ["line1","line2","line3","line4","line5","line6"]

    for c in parts.columns:
        parts[c] = pd.to_numeric(parts[c], errors="coerce")

    parts = parts.dropna()
    df = df.loc[parts.index]

    print("Valid rows:", len(df))
    return parts.astype(int).values

def detect_cycles(labels, max_cycle_len=6):
    cycles = {}
    n = len(labels)

    for L in range(2, max_cycle_len+1):
        seen = {}
        for i in range(n - L):
            seq = tuple(labels[i:i+L])
            if seq in seen:
                seen[seq] += 1
            else:
                seen[seq] = 1

        repeated = {seq:count for seq,count in seen.items() if count > 1}
        if repeated:
            cycles[L] = repeated

    return cycles

# ============================
# LOAD HP DATASET
# ============================
X = load_and_expand("hexagramdataset1.csv")

k = 20
print("\nUsing k =", k, "clusters.\n")

# ============================
# CLUSTER HP DATASET
# ============================
print("Clustering HP dataset...")
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
labels = kmeans.labels_

# ============================
# DETECT BASIN CYCLES
# ============================
print("\nDetecting basin cycles...")
cycles = detect_cycles(labels, max_cycle_len=6)

# ============================
# WRITE RESULTS TO FILE
# ============================
output_path = "basin_cycles_output.txt"
with open(output_path, "w") as f:
    f.write("BASIN-CYCLE DETECTION RESULTS\n")
    f.write("=============================\n\n")

    if not cycles:
        f.write("No repeated cycles found.\n")
    else:
        for L, seqs in cycles.items():
            f.write(f"\n--- Cycle length = {L} ---\n")
            for seq, count in seqs.items():
                f.write(f"Cycle {seq}  |  repeats = {count}\n")

print(f"\nDone. Output written to {output_path}\n")
