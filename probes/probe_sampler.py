import time
import psutil
import platform
import os
import csv
from datetime import datetime, timezone
import random

process_start_unix = time.time()
system_boot_time_unix = psutil.boot_time()
sampling_interval_seconds = 1.0

output_file = "structured_output.csv"

header = [
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
]

if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def generate_hexagram():
    lines = [random.choice([7, 8]) for _ in range(6)]
    changing = random.choice(["", "6", "2-6", "3", "4-5"])
    return "-".join(str(x) for x in lines), changing

while True:
    timestamp_unix = time.time()
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    hexagram_lines, changing_lines = generate_hexagram()

    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage(os.getcwd()).percent  # FIXED FOR WINDOWS

    row = [
        timestamp_iso,
        timestamp_unix,
        hexagram_lines,
        changing_lines,
        "",
        platform.node(),
        platform.platform(),
        platform.python_version(),
        os.path.basename(__file__),
        os.getcwd(),
        sampling_interval_seconds,
        process_start_unix,
        timestamp_unix - process_start_unix,
        cpu_percent,
        ram_percent,
        disk_percent,
        system_boot_time_unix
    ]

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    time.sleep(sampling_interval_seconds)
