import csv
import os
import math

INPUT_FILE = "raw_temporal_dataset.csv"
OUTPUT_FILE = "temporal_analysis_dataset.csv"

def read_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def f(r, key):
    try:
        return float(r[key])
    except:
        return float("nan")

def analyze(rows):
    out = []

    prev_mean = None
    prev_var = None
    prev_auto = None
    prev_centroid = None

    # rolling stability window
    window = []
    WINDOW = 30

    for r in rows:
        ts_iso = r["timestamp_iso"]
        ts_unix = f(r, "timestamp_unix")
        mean_val = f(r, "mean")
        var_val = f(r, "variance")
        auto_val = f(r, "autocorrelation")
        centroid_val = f(r, "spectral_centroid")

        # deltas
        if prev_mean is None:
            mean_delta = float("nan")
            var_delta = float("nan")
            auto_delta = float("nan")
            centroid_delta = float("nan")
        else:
            mean_delta = mean_val - prev_mean
            var_delta = var_val - prev_var
            auto_delta = auto_val - prev_auto
            centroid_delta = centroid_val - prev_centroid

        # drift = absolute delta
        auto_drift = abs(auto_delta) if not math.isnan(auto_delta) else float("nan")
        centroid_drift = abs(centroid_delta) if not math.isnan(centroid_delta) else float("nan")
        var_drift = abs(var_delta) if not math.isnan(var_delta) else float("nan")

        # variance evolution (relative change)
        if prev_var is None or prev_var == 0 or math.isnan(prev_var):
            var_rel = float("nan")
        else:
            var_rel = (var_val - prev_var) / prev_var

        # stability score (inverse drift)
        window.append({
            "m": abs(mean_delta) if not math.isnan(mean_delta) else 0.0,
            "v": abs(var_delta) if not math.isnan(var_delta) else 0.0,
            "a": abs(auto_delta) if not math.isnan(auto_delta) else 0.0,
            "c": abs(centroid_delta) if not math.isnan(centroid_delta) else 0.0,
        })
        if len(window) > WINDOW:
            window.pop(0)

        if len(window) < WINDOW:
            stability = float("nan")
        else:
            m = sum(w["m"] for w in window) / WINDOW
            v = sum(w["v"] for w in window) / WINDOW
            a = sum(w["a"] for w in window) / WINDOW
            c = sum(w["c"] for w in window) / WINDOW
            stability = 1.0 / (1.0 + m + v + a + c)

        out.append({
            "timestamp_iso": ts_iso,
            "timestamp_unix": ts_unix,
            "mean": mean_val,
            "variance": var_val,
            "autocorrelation": auto_val,
            "spectral_centroid": centroid_val,
            "mean_delta": mean_delta,
            "variance_delta": var_delta,
            "autocorrelation_delta": auto_delta,
            "spectral_centroid_delta": centroid_delta,
            "autocorrelation_drift": auto_drift,
            "spectral_centroid_drift": centroid_drift,
            "variance_drift": var_drift,
            "variance_relative_change": var_rel,
            "temporal_stability_score": stability,
        })

        prev_mean = mean_val
        prev_var = var_val
        prev_auto = auto_val
        prev_centroid = centroid_val

    return out

def write_rows(path, rows):
    fields = [
        "timestamp_iso",
        "timestamp_unix",
        "mean",
        "variance",
        "autocorrelation",
        "spectral_centroid",
        "mean_delta",
        "variance_delta",
        "autocorrelation_delta",
        "spectral_centroid_delta",
        "autocorrelation_drift",
        "spectral_centroid_drift",
        "variance_drift",
        "variance_relative_change",
        "temporal_stability_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    rows = read_rows(INPUT_FILE)
    out = analyze(rows)
    write_rows(OUTPUT_FILE, out)
    print("Analysis complete:", os.path.abspath(OUTPUT_FILE))

if __name__ == "__main__":
    main()
