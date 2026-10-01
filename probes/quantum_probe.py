import csv
import time
import requests
import psutil
import os
from datetime import datetime, timezone

# ============================
# QRNGAPI CONFIG
# ============================
API_KEY = "qnrk_e3c53dc13eb0fbfd49f21025c26564196ff4d8cc73869ac9"

# Correct endpoint format
QRNG_URL = f"https://qrngapi.com/api/random/uint8?api_key={API_KEY}&size=18"

OUTPUT_FILE = "structured_output_quantum.csv"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (QuantumProbe/1.0)"
})

# ============================
# RESILIENT QRNG FETCHER
# ============================
def get_qrng_block(num_bytes=18):
    backoff = 1.0
    while True:
        try:
            r = session.get(QRNG_URL, timeout=(3.05, 5))

            # Try JSON
            try:
                data = r.json()
            except Exception:
                print(f"Non-JSON response. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 16.0)
                continue

            # Validate JSON
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) >= num_bytes:
                return data["data"][:num_bytes]

            print(f"Malformed JSON. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 16.0)

        except Exception as e:
            print(f"Network error ({type(e).__name__}). Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 16.0)

# ============================
# HEXAGRAM GENERATION
# ============================
def qrng_hexagram():
    raw_18_bytes = get_qrng_block(18)

    lines = []
    raw_bits = []

    for i in range(6):
        b1, b2, b3 = raw_18_bytes[i*3 : (i+1)*3]
        c1 = 2 if b1 < 128 else 3
        c2 = 2 if b2 < 128 else 3
        c3 = 2 if b3 < 128 else 3

        line_value = c1 + c2 + c3
        lines.append(line_value)
        raw_bits.append((b1, b2, b3))

    return lines, raw_bits

# ============================
# SYSTEM INFO
# ============================
def get_system_boot_time_unix():
    return psutil.boot_time()

def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    return cpu, ram, disk

# ============================
# CSV INIT
# ============================
def init_csv(path):
    exists = os.path.isfile(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not exists:
        writer.writerow([
            "timestamp_iso",
            "timestamp_unix",
            "line1",
            "line2",
            "line3",
            "line4",
            "line5",
            "line6",
            "changing_lines",
            "cpu_percent",
            "ram_percent",
            "disk_percent",
            "time_since_process_start_seconds",
            "system_boot_time_unix",
            "num_changing",
            "cluster"
        ])
    return f, writer

# ============================
# MAIN LOOP (FIXED 1-SECOND CADENCE)
# ============================
def main():
    start_time = time.time()
    boot_time = get_system_boot_time_unix()
    f, writer = init_csv(OUTPUT_FILE)

    print("Quantum probe (QRNGAPI.com, Fixed Interval) running. Writing to:", OUTPUT_FILE)
    print("Press Ctrl+C to stop.")

    next_sample_time = start_time

    try:
        while True:
            now = time.time()
            sleep_time = next_sample_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_start = time.time()
            next_sample_time = loop_start + 1.0

            ts_iso = datetime.fromtimestamp(loop_start, timezone.utc).isoformat()
            ts_unix = int(loop_start)

            lines, raw_bits = qrng_hexagram()
            line1, line2, line3, line4, line5, line6 = lines

            changing = [str(i) for i, v in enumerate(lines, start=1) if v in (6, 9)]
            changing_str = ",".join(changing) if changing else ""
            num_changing = len(changing)

            cpu, ram, disk = get_system_stats()
            t_since_start = loop_start - start_time

            writer.writerow([
                ts_iso,
                ts_unix,
                line1,
                line2,
                line3,
                line4,
                line5,
                line6,
                changing_str,
                cpu,
                ram,
                disk,
                t_since_start,
                int(boot_time),
                num_changing,
                -1
            ])

            f.flush()

    except KeyboardInterrupt:
        print("\nStopped.")
        f.close()

if __name__ == "__main__":
    main()
