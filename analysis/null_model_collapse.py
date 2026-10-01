import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("structured_output_clustered.csv")

X = df[["line1","line2","line3","line4","line5","line6"]].values

def build_markov(labels):
    uniq = sorted(np.unique(labels))
    map_ = {u:i for i,u in enumerate(uniq)}
    seq = np.array([map_[l] for l in labels])
    n = len(uniq)

    T = np.zeros((n,n))
    for i in range(len(seq)-1):
        T[seq[i], seq[i+1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    P = T / row_sums

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:,idx])
    stat = stat / stat.sum()

    return uniq, stat

def print_attractors(name, uniq, stat):
    print("\n==============================")
    print(f"{name} ATTRACTORS")
    print("==============================")
    idxs = np.argsort(stat)[::-1][:10]
    for i in idxs:
        print(f"  State {uniq[i]}: {stat[i]:.6f}")

def drift(a,b):
    L = max(len(a),len(b))
    a2 = np.pad(a,(0,L-len(a)))
    b2 = np.pad(b,(0,L-len(b)))
    return np.sum(np.abs(a2-b2))

# REAL DATA
kmeans = KMeans(n_clusters=20, random_state=42)
real_labels = kmeans.fit_predict(X)
uniq_real, stat_real = build_markov(real_labels)

print_attractors("REAL DATA", uniq_real, stat_real)

# SHUFFLE NULL
X_shuffle = X.copy()
np.random.shuffle(X_shuffle)
shuffle_labels = kmeans.fit_predict(X_shuffle)
uniq_shuf, stat_shuf = build_markov(shuffle_labels)

print_attractors("SHUFFLE NULL", uniq_shuf, stat_shuf)
print("\nDrift REAL ↔ SHUFFLE:", drift(stat_real, stat_shuf))

# PERMUTATION NULL
perm_idx = np.random.permutation(len(X))
X_perm = X[perm_idx]
perm_labels = kmeans.fit_predict(X_perm)
uniq_perm, stat_perm = build_markov(perm_labels)

print_attractors("PERMUTATION NULL", uniq_perm, stat_perm)
print("\nDrift REAL ↔ PERMUTATION:", drift(stat_real, stat_perm))

# RANDOM HEXAGRAM NULL
rand = np.random.choice([6,7,8,9], size=X.shape)
rand_labels = kmeans.fit_predict(rand)
uniq_rand, stat_rand = build_markov(rand_labels)

print_attractors("RANDOM HEXAGRAM NULL", uniq_rand, stat_rand)
print("\nDrift REAL ↔ RANDOM:", drift(stat_real, stat_rand))

print("\nDone.")
