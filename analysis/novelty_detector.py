import time
import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\PSNet Cannabis\Downloads\RTDx64\raw_temporal_dataset_x64.csv"

WINDOW = 128
SLEEP = 1.0

FEATURES = [
    "mean",
    "variance",
    "autocorrelation",
    "spectral_energy",
    "spectral_centroid"
]

def safe_read(path):
    try:
        return pd.read_csv(path)
    except:
        return None

def compute_curve_novelty(df):
    if df is None or len(df) < WINDOW + 3:
        return None, None

    recent = df.tail(WINDOW)
    novelty_scores = []

    for col in FEATURES:
        y = recent[col].values

        # First derivative
        d1 = np.diff(y)

        # Second derivative (curvature)
        d2 = np.diff(d1)

        # Third derivative (jerk)
        d3 = np.diff(d2)

        # Novelty = combined magnitude of curvature + jerk
        score = np.mean(np.abs(d2)) + 0.5 * np.mean(np.abs(d3))
        novelty_scores.append(score)

    novelty_raw = np.mean(novelty_scores)

    # Normalize using recent typical curvature
    baseline = np.std(novelty_scores) + 1e-9
    novelty_norm = novelty_raw / baseline

    return novelty_norm, recent["timestamp_iso"].iloc[-1]

def classify(n):
    if n < 0.8:
        return "HABIT"
    elif n < 1.2:
        return "TRANSITION"
    else:
        return "NOVELTY"

def main():
    print("[INIT] Novelty/Habit detector running (curve mode)...")

    last_len = 0

    while True:
        df = safe_read(CSV_PATH)
        if df is None:
            time.sleep(SLEEP)
            continue

        if len(df) == last_len:
            time.sleep(SLEEP)
            continue

        last_len = len(df)

        novelty, ts = compute_curve_novelty(df)
        if novelty is None:
            continue

        state = classify(novelty)
        print(f"{ts} | novelty={novelty:.3f} | {state}")

        time.sleep(SLEEP)

if __name__ == "__main__":
    main()
