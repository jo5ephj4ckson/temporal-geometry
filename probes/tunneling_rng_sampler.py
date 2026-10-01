import csv
import os
import time
import platform
import psutil
import sounddevice as sd
import numpy as np
import hashlib
from datetime import datetime, UTC

DATA_FILE = "tunneling_hexagram_dataset.csv"

# --- system state (static per run) ---
PROCESS_START_UNIX = time.time()
MACHINE_NAME = platform.node()
OS_INFO = f"{platform.system()} {platform.release()}"
PYTHON_VERSION = platform.python_version()
SCRIPT_NAME = os.path.basename(__file__) if "__file__" in globals() else "interactive"
WORKING_DIR = os.getcwd()

# --- sampling interval ---
SAMPLING_INTERVAL_SECONDS = 1  # adjust as needed

# --- entropy capture from LED tunneling probe ---
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
CHANNELS = 1

def get_entropy_block():
    """Capture raw audio bytes from LED tunneling probe."""
    data = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return data.tobytes()

def entropy_to_bits(entropy_bytes, bits=48):
    """Hash entropy and return 'bits' random bits."""
    h = hashlib.sha512(entropy_bytes).digest()
    bitstring = ''.join(f'{byte:08b}' for byte in h)
    return bitstring[:bits]

def bits_to_int(bitstring):
    return int(bitstring, 2)

# --- hexagram generation using tunneling entropy ---
def generate_hexagram_from_entropy():
    """
    Generate 6 lines using LED tunneling entropy.
    Each line is derived from 8 bits of entropy.
    """
    entropy = get_entropy_block()
    bits = entropy_to_bits(entropy, bits=48)  # 6 lines * 8 bits each

    lines = []
    for i in range(6):
        segment = bits[i*8:(i+1)*8]
        val = bits_to_int(segment)

        # Map entropy to hexagram line values (6,7,8,9)
        line_value = [6, 7, 8, 9][val % 4]
        changing = (line_value == 6 or line_value == 9)
        kind = "solid" if line_value in (7, 9) else "broken"
        age = "old_yang" if line_value == 9 else \
              "old_yin" if line_value == 6 else \
              "young_yang" if line_value == 7 else \
              "young_yin"

        lines.append((line_value, age, kind, changing))

    return lines

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

    lines = generate_hexagram_from_entropy()
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

    print("Recorded hexagram:")
    print("  timestamp:", ts_iso)
    print("  lines:", hexagram_str)
    print("  changing:", changing_str)
    print("  cpu:", cpu_percent)
    print("  ram:", ram_percent)
    print("  disk:", disk_percent)

# --- run loop ---
print("Starting LED tunneling RNG hexagram sampler...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds")

while True:
    record_hexagram("tunneling RNG sample")
    time.sleep(SAMPLING_INTERVAL_SECONDS)
