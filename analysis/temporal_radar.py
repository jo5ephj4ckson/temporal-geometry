import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "raw_temporal_dataset_x64.csv"
POLL_INTERVAL_SEC = 0.25
WINDOW_SIZE = 512  # number of samples to show in radar

def load_last_n(path, n):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df["mean"].tail(n).values
    except Exception:
        return None

def main():
    plt.ion()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    while True:
        try:
            data = load_last_n(CSV_PATH, WINDOW_SIZE)
            if data is None or len(data) < 4:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Normalize mean values for plotting
            arr = np.array(data)
            arr_norm = arr - np.min(arr)
            if np.max(arr_norm) > 0:
                arr_norm = arr_norm / np.max(arr_norm)

            # Radar angles
            theta = np.linspace(0, 2 * np.pi, len(arr_norm), endpoint=False)
            theta = np.append(theta, theta[0])
            r = np.append(arr_norm, arr_norm[0])

            ax.clear()
            ax.set_title("Real‑Time Temporal Radar (mean drift)", va='bottom')
            ax.set_ylim(0, 1.0)

            ax.plot(theta, r, color='cyan', linewidth=2)
            ax.fill(theta, r, color='cyan', alpha=0.25)

            plt.draw()
            plt.pause(0.001)

            time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(POLL_INTERVAL_SEC)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
