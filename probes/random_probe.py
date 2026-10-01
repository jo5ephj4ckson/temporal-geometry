import csv
import time
import requests
import psutil
import os
from datetime import datetime, timezone
import base64

API_KEY = "e5f9a0fd-55ca-4b5d-acc1-8c09fff31dfa"
RND_URL = "https://api.random.org/json-rpc/4/invoke"
OUTPUT_FILE = "structured_output_randomorg.csv"

SAMPLE_INTERVAL = 1.12  # safe rate for your quota

session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "User-Agent": "PSNet-Hexagram-Probe/1.0"
})

def get_random_bytes():
    payload = {
        "jsonrpc": "2.0",
        "method": "generateBlobs",
        "params": {
            "apiKey": API_KEY,
            "n": 1,
            "size": 144  # 18 bytes = 144 bits
        },
        "id": 1
    }

    while True:
        try:
            r = session.post(RND_URL, json=payload, timeout=(3.05, 5))
            data = r.json()

            # If Random.org returns an error, show it once and retry
            if "error" in data:
                print(f"Random.org error: {data['error']}")
                time.sleep(1)
                continue

            if "result" in data and "random" in data["result"]:
                blob_b64 = data["result"]["random"]["data"][0]
                raw_bytes = base64.b64decode(blob_b64)
                if len(raw_bytes) == 18:
                    return list(raw_bytes)

            print("Malformed data length, retrying...")
            time.sleep(1)

        except Exception as e:
            print(f"Network error ({type(e).__name__}), retrying...")
            time.sleep(1)

def hexagram_from_bytes(raw_18_bytes):
    lines = []
    for i in range(6):
        b1, b2, b3 = raw_18_bytes[i*3:(i+1)*3]
        c1 = 2 if b1 < 128 else 3
        c2 = 2 if b2 < 128 else 3
        c3 = 2 if b3 < 128 else 3
        lines.append(c1 + c2 + c3)
    return lines

def get_system_stats():
    return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent, psutil.disk_usage("/").percent

def init_csv(path):
    exists = os.path.isfile(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not exists:
        writer.writerow([
            "timestamp_iso","timestamp_unix",
            "line1","line2","line3","line4","line5","line6",
            "changing_lines","cpu_percent","ram_percent","disk_percent",
            "time_since_start_seconds","num_changing","cluster"
        ])
    return f, writer

def main():
    start_time = time.time()
    f, writer = init_csv(OUTPUT_FILE)
    print("Random.org probe running (1 sample per request, 18 bytes, 1.12s interval).")

    next_sample_time = start_time

    try:
        while True:
            sleep_time = next_sample_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            loop_start = time.time()
            next_sample_time = loop_start + SAMPLE_INTERVAL

            ts_iso = datetime.fromtimestamp(loop_start, timezone.utc).isoformat()
            ts_unix = int(loop_start)

            raw_bytes = get_random_bytes()
            lines = hexagram_from_bytes(raw_bytes)
            changing = [str(i) for i, v in enumerate(lines, start=1) if v in (6, 9)]
            cpu, ram, disk = get_system_stats()

            writer.writerow([
                ts_iso, ts_unix, *lines,
                ",".join(changing), cpu, ram, disk,
                loop_start - start_time, len(changing), -1
            ])
            f.flush()

    except KeyboardInterrupt:
        print("\nStopped.")
        f.close()

if __name__ == "__main__":
    main()
