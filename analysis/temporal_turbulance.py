#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CSV_FILE = "raw_temporal_dataset_x64.csv"
TIME_COL = "timestamp_unix"
VALUE_COL = "mean"

N_MICRO_STATES = 32
N_REGIMES = 3
WINDOW_SIZE = 100
TURB_WINDOW = 200  # window for turbulence metrics


def build_micro_states(series, n_states):
    quantiles = np.linspace(0, 1, n_states + 1)
    bins = np.unique(np.quantile(series, quantiles))
    if len(bins) <= 2:
        bins = np.linspace(series.min(), series.max(), n_states + 1)
    states = np.digitize(series, bins[1:-1])
    return states, bins


def markov_matrix(states, n_states):
    M = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        s, t = states[i], states[i + 1]
        if 0 <= s < n_states and 0 <= t < n_states:
            M[s, t] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


def markov_feature_windows(states, n_states, window_size):
    feats = []
    idxs = []
    for i in range(0, len(states) - window_size):
        w = states[i:i + window_size]
        M = markov_matrix(w, n_states)
        feats.append(M.flatten())
        idxs.append(i)
    return np.array(feats), np.array(idxs)


def cluster_regimes(features, n_regimes):
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    return km.fit_predict(features)


def simulate_regimes(n_steps, T):
    regimes = np.zeros(n_steps, dtype=int)
    for i in range(1, n_steps):
        regimes[i] = np.random.choice([0, 1, 2], p=T[regimes[i - 1]])
    return regimes


def regime_transition_matrix(labels, n_regimes):
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        T[labels[i], labels[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def regime_stats(labels, idxs, series, window_size, n_regimes):
    stats = {}
    for r in range(n_regimes):
        mask = labels == r
        if not np.any(mask):
            continue
        starts = idxs[mask]
        means = []
        vars_ = []
        for s in starts:
            seg = series[s:s + window_size]
            means.append(seg.mean())
            vars_.append(seg.var())
        stats[r] = {"mean": float(np.mean(means)),
                    "var": float(np.mean(vars_))}
    return stats


def regime_entropy(reg_seq, n_regimes):
    counts = np.bincount(reg_seq, minlength=n_regimes)
    p = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def turbulence_metrics(regimes, values, window, n_regimes):
    rows = []
    for i in range(0, len(regimes) - window):
        r_win = regimes[i:i + window]
        v_win = values[i:i + window]

        # regime switches
        switches = np.sum(r_win[1:] != r_win[:-1])

        # regime entropy
        ent = regime_entropy(r_win, n_regimes)

        # value volatility (std)
        vol = float(np.std(v_win))

        # combined turbulence score (simple linear combo)
        score = switches + ent * 10.0 + vol * 1e7

        rows.append({
            "start_index": i,
            "end_index": i + window,
            "regime_switches": int(switches),
            "regime_entropy": float(ent),
            "value_volatility": vol,
            "turbulence_score": float(score),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV_FILE).sort_values(TIME_COL).reset_index(drop=True)
    series = df[VALUE_COL].values
    time = df[TIME_COL].values

    print(f"Loaded {len(series)} samples.")

    # Build manifold
    micro_states, bins = build_micro_states(series, N_MICRO_STATES)
    feats, idxs = markov_feature_windows(micro_states, N_MICRO_STATES, WINDOW_SIZE)
    regimes = cluster_regimes(feats, N_REGIMES)

    # Align regime labels to series length (approximate)
    reg_full = np.zeros(len(series), dtype=int)
    for i, start in enumerate(idxs):
        end = min(start + WINDOW_SIZE, len(series))
        reg_full[start:end] = regimes[i]

    T = regime_transition_matrix(regimes, N_REGIMES)
    stats = regime_stats(regimes, idxs, series, WINDOW_SIZE, N_REGIMES)

    print("\nRegime transition matrix:")
    print(T)
    print("\nRegime stats:")
    for r, s in stats.items():
        print(f"Regime {r}: mean={s['mean']:.6e}, var={s['var']:.6e}")

    # Turbulence detection
    turb_df = turbulence_metrics(reg_full, series, TURB_WINDOW, N_REGIMES)

    print("\nTop 10 highest-turbulence windows:")
    top = turb_df.sort_values("turbulence_score", ascending=False).head(10)
    print(top)

    # Optional: save full turbulence table
    turb_df.to_csv("temporal_turbulence_windows.csv", index=False)
    print("\nSaved turbulence windows to temporal_turbulence_windows.csv")


if __name__ == "__main__":
    main()
