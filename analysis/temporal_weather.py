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

# ALIGN WINDOWS SO MERGE NEVER FAILS
TURB_WINDOW = 200
PRESSURE_WINDOW = 200


# ---------------------------------------------------------
# Microstate + Markov utilities
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Regime utilities
# ---------------------------------------------------------

def regime_entropy(reg_seq, n_regimes):
    counts = np.bincount(reg_seq, minlength=n_regimes)
    p = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


# ---------------------------------------------------------
# 1. TEMPORAL TURBULENCE
# ---------------------------------------------------------

def turbulence_metrics(regimes, values, window, n_regimes):
    rows = []
    for i in range(0, len(regimes) - window):
        r_win = regimes[i:i + window]
        v_win = values[i:i + window]

        switches = np.sum(r_win[1:] != r_win[:-1])
        ent = regime_entropy(r_win, n_regimes)
        vol = float(np.std(v_win))

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


# ---------------------------------------------------------
# 2. TEMPORAL PRESSURE
# ---------------------------------------------------------

def temporal_pressure(values, regimes, window):
    rows = []
    for i in range(0, len(values) - window):
        v_win = values[i:i + window]
        r_win = regimes[i:i + window]

        vol = float(np.std(v_win))
        ent = regime_entropy(r_win, N_REGIMES)

        pressure = vol * 1e7 + ent * 10.0

        rows.append({
            "start_index": i,
            "end_index": i + window,
            "pressure": pressure,
            "volatility": vol,
            "entropy": ent
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. TEMPORAL FRONT TRACKING (percentile-based)
# ---------------------------------------------------------

def detect_fronts(turb_df, pressure_df):
    merged = turb_df.merge(pressure_df, on=["start_index", "end_index"])

    if merged.empty:
        print("\nWARNING: merged DataFrame is empty — no aligned windows.")
        return merged, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged["front_score"] = merged["turbulence_score"] + merged["pressure"]

    strong_thresh = np.percentile(merged["front_score"], 95)
    moderate_thresh = np.percentile(merged["front_score"], 90)
    weak_thresh = np.percentile(merged["front_score"], 85)

    strong_fronts = merged[merged["front_score"] >= strong_thresh]
    moderate_fronts = merged[(merged["front_score"] >= moderate_thresh) &
                             (merged["front_score"] < strong_thresh)]
    weak_fronts = merged[(merged["front_score"] >= weak_thresh) &
                         (merged["front_score"] < moderate_thresh)]

    return merged, strong_fronts, moderate_fronts, weak_fronts


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    df = pd.read_csv(CSV_FILE).sort_values(TIME_COL).reset_index(drop=True)
    series = df[VALUE_COL].values

    print(f"Loaded {len(series)} samples.")

    micro_states, bins = build_micro_states(series, N_MICRO_STATES)
    feats, idxs = markov_feature_windows(micro_states, N_MICRO_STATES, WINDOW_SIZE)
    regimes = cluster_regimes(feats, N_REGIMES)

    reg_full = np.zeros(len(series), dtype=int)
    for i, start in enumerate(idxs):
        end = min(start + WINDOW_SIZE, len(series))
        reg_full[start:end] = regimes[i]

    turb_df = turbulence_metrics(reg_full, series, TURB_WINDOW, N_REGIMES)
    pressure_df = temporal_pressure(series, reg_full, PRESSURE_WINDOW)

    weather_map, strong, moderate, weak = detect_fronts(turb_df, pressure_df)

    print("\nStrong fronts:")
    print(strong.head(10))

    print("\nModerate fronts:")
    print(moderate.head(10))

    print("\nWeak fronts:")
    print(weak.head(10))

    weather_map.to_csv("temporal_weather_map.csv", index=False)
    strong.to_csv("temporal_fronts_strong.csv", index=False)
    moderate.to_csv("temporal_fronts_moderate.csv", index=False)
    weak.to_csv("temporal_fronts_weak.csv", index=False)

    print("\nSaved temporal weather map and front lists.")


if __name__ == "__main__":
    main()
