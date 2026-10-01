import csv
import os
import time
import platform
import psutil
import sounddevice as sd
import hashlib
from datetime import datetime, UTC

DATA_FILE = "16bit_tunneling_dataset.csv"

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

def entropy_to_uint16(entropy_bytes):
    """Hash entropy and return a 16-bit integer (0..65535)."""
    h = hashlib.sha512(entropy_bytes).digest()
    bitstring = ''.join(f'{byte:08b}' for byte in h)
    return int(bitstring[:16], 2)

# --- dataset handling ---
def init_dataset():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_iso",
                "timestamp_unix",
                "symbol_current",
                "symbol_previous",
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

_last_symbol = None

def record_symbol(notes="tunneling 16-bit sample"):
    global _last_symbol
    init_dataset()

    entropy_bytes = get_entropy_block()
    symbol_current = entropy_to_uint16(entropy_bytes)
    symbol_previous = _last_symbol if _last_symbol is not None else "none"
    _last_symbol = symbol_current

    ts = datetime.now(UTC)
    ts_iso = ts.isoformat()
    ts_unix = ts.timestamp()

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
            symbol_current,
            symbol_previous,
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

    print("Recorded 16-bit symbol:")
    print("  timestamp:", ts_iso)
    print("  symbol:", symbol_current, "prev:", symbol_previous)
    print("  cpu:", cpu_percent)
    print("  ram:", ram_percent)
    print("  disk:", disk_percent)

# --- run loop ---
print("Starting LED tunneling RNG 16-bit sampler...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds")

while True:
    record_symbol()
    time.sleep(SAMPLING_INTERVAL_SECONDS)
