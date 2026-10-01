import random
import csv
from datetime import datetime, UTC
import os
import time
import platform
import sys
import psutil

DATA_FILE = "inverted_hexagram_dataset.csv"

# --- system state (static per run) ---
PROCESS_START_UNIX = time.time()
MACHINE_NAME = platform.node()
OS_INFO = f"{platform.system()} {platform.release()}"
PYTHON_VERSION = platform.python_version()
SCRIPT_NAME = os.path.basename(__file__) if "__file__" in globals() else "interactive"
WORKING_DIR = os.getcwd()

# --- sampling interval ---
SAMPLING_INTERVAL_SECONDS = 1  # change to 60 for slow dataset

# --- hexagram generation ---
def toss_line():
    coins = [random.choice([2, 3]) for _ in range(3)]
    total = sum(coins)

    if total == 6:
        return 6, "old_yin", "broken", True
    elif total == 7:
        return 7, "young_yang", "solid", False
    elif total == 8:
        return 8, "young_yin", "broken", False
    elif total == 9:
        return 9, "old_yang", "solid", True
    else:
        raise ValueError("Invalid coin sum")

def generate_hexagram():
    """
    ORIGINAL: bottom line first → top line last
    INVERTED: top line first → bottom line last
    So we generate normally, then reverse the 6-line list.
    """
    lines = [toss_line() for _ in range(6)]
    return lines[::-1]   # <-- INVERSION RIGHT HERE

# --- dataset handling ---
def init_dataset():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_iso",
                "timestamp_unix",
                "hexagram_lines",
                "changing_lines",
                "notes",
                "machine_name",
                "os_info",
                "python_version",
                "script_name",
                "working_dir",
                "sampling_interval_seconds",
                "process_start_unix",
                "time_since_process_start_seconds",
                "cpu_percent",
                "ram_percent",
                "disk_percent",
                "system_boot_time_unix"
            ])

def hexagram_to_strings(lines):
    hexagram_str = "-".join(str(v) for (v, _, _, _) in lines)
    changing_str = "-".join(
        str(i + 1) for i, (_, _, _, changing) in enumerate(lines) if changing
    ) or "none"
    return hexagram_str, changing_str

def record_hexagram(notes=""):
    init_dataset()

    lines = generate_hexagram()
    ts = datetime.now(UTC)
    ts_iso = ts.isoformat()
    ts_unix = ts.timestamp()

    hexagram_str, changing_str = hexagram_to_strings(lines)

    # --- dynamic system state ---
    time_since_start = time.time() - PROCESS_START_UNIX
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage(WORKING_DIR).percent
    system_boot_time_unix = psutil.boot_time()

    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts_iso,
            ts_unix,
            hexagram_str,
            changing_str,
            notes,
            MACHINE_NAME,
            OS_INFO,
            PYTHON_VERSION,
            SCRIPT_NAME,
            WORKING_DIR,
            SAMPLING_INTERVAL_SECONDS,
            PROCESS_START_UNIX,
            time_since_start,
            cpu_percent,
            ram_percent,
            disk_percent,
            system_boot_time_unix
        ])

    print("Recorded INVERTED hexagram:")
    print("  timestamp:", ts_iso)
    print("  lines:", hexagram_str)
    print("  changing:", changing_str)
    print("  cpu:", cpu_percent)
    print("  ram:", ram_percent)
    print("  disk:", disk_percent)

# --- run loop ---
print("Starting INVERTED hexagram sampler...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds")

while True:
    record_hexagram("auto sample")
    time.sleep(SAMPLING_INTERVAL_SECONDS)
