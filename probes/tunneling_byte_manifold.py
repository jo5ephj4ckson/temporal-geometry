import csv
import os
import time
import platform
import psutil
import sounddevice as sd
import numpy as np
import hashlib
from datetime import datetime, UTC

DATA_FILE = "tunneling_byte_manifold_dataset.csv"

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

def entropy_to_bits(entropy_bytes, bits=8):
    """Hash entropy and return 'bits' random bits."""
    h = hashlib.sha512(entropy_bytes).digest()
    bitstring = ''.join(f'{byte:08b}' for byte in h)
    return bitstring[:bits]

def bits_to_int(bitstring):
    return int(bitstring, 2)

# --- byte-manifold symbol generation using tunneling entropy ---
def generate_symbol_from_entropy():
    """
    Generate a single 8-bit symbol (0-255) from LED tunneling entropy.
    """
    entropy = get_entropy_block()
    bits = entropy_to_bits(entropy, bits=8)
    value = bits_to_int(bits)
    return value

def hamming_distance(a: int, b: int) -> int:
    """
    Hamming distance between two 8-bit integers.
    """
    x = a ^ b
    return bin(x).count("1")

# --- dataset handling ---
def init_dataset():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_iso",
                "timestamp_unix",
                "symbol_current",
                "symbol_next",
                "symbol_current_bits",
                "symbol_next_bits",
                "bit_change_count",
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

def symbol_to_bitstring(value: int) -> str:
    """
    Convert 8-bit integer to bitstring 'b7b6...b0'.
    """
    return "".join("1" if (value & (1 << i)) else "0" for i in range(7, -1, -1))

_last_symbol = None

def record_symbol(notes: str = ""):
    global _last_symbol

    init_dataset()

    current_symbol = generate_symbol_from_entropy()

    if _last_symbol is None:
        bit_change_count = 0
        next_symbol = current_symbol
    else:
        next_symbol = current_symbol
        bit_change_count = hamming_distance(_last_symbol, next_symbol)

    ts = datetime.now(UTC)
    ts_iso = ts.isoformat()
    ts_unix = ts.timestamp()

    current_bits = symbol_to_bitstring(current_symbol)
    next_bits = symbol_to_bitstring(next_symbol)

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
            current_symbol,
            next_symbol,
            current_bits,
            next_bits,
            bit_change_count,
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

    print("Recorded tunneling symbol:")
    print("  timestamp:", ts_iso)
    print("  current_symbol:", current_symbol, "bits:", current_bits)
    print("  next_symbol:", next_symbol, "bits:", next_bits)
    print("  bit_change_count:", bit_change_count)
    print("  cpu:", cpu_percent)
    print("  ram:", ram_percent)
    print("  disk:", disk_percent)

    _last_symbol = current_symbol

# --- run loop ---
print("Starting LED tunneling RNG 8-bit manifold sampler...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds")

while True:
    record_symbol("tunneling RNG sample")
    time.sleep(SAMPLING_INTERVAL_SECONDS)
