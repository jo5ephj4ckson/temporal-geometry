import csv
import os
import time
import platform
import psutil
import sounddevice as sd
import numpy as np
from datetime import datetime, UTC

# -----------------------------
# CONFIG (64-BIT PROBE)
# -----------------------------
# Use the device index that gave you real variance (12 or 18)
sd.default.device = 12          # change to 18 if needed

SAMPLE_RATE = 48000             # confirmed supported
CHUNK_SIZE = 4096               # slightly larger block for finer temporal texture
CHANNELS = 1

# 64-bit audio capture
AUDIO_DTYPE = 'float64'         # raw ADC values as 64-bit float

OUTPUT_FILE = "raw_temporal_dataset_x64.csv"
SAMPLING_INTERVAL_SECONDS = 1   # seconds between samples

# -----------------------------
# SYSTEM METADATA
# -----------------------------
PROCESS_START_UNIX = time.time()
MACHINE_NAME = platform.node()
OS_INFO = f"{platform.system()} {platform.release()}"
PYTHON_VERSION = platform.python_version()
SCRIPT_NAME = os.path.basename(__file__) if "__file__" in globals() else "interactive"
WORKING_DIR = os.getcwd()

# -----------------------------
# DATASET INIT
# -----------------------------
def init_dataset():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_iso",
                "timestamp_unix",
                "mean",
                "variance",
                "min",
                "max",
                "autocorrelation",
                "spectral_energy",
                "spectral_centroid",
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

# -----------------------------
# TEMPORAL FEATURE FUNCTIONS
# -----------------------------
def capture_block():
    try:
        block = sd.rec(CHUNK_SIZE,
                       samplerate=SAMPLE_RATE,
                       channels=CHANNELS,
                       dtype=AUDIO_DTYPE)
        sd.wait()
        # ensure 1D float64 array
        return block.flatten().astype(np.float64)
    except Exception as e:
        print("ERROR capturing audio:", e)
        return np.zeros(CHUNK_SIZE, dtype=np.float64)

def block_mean(block):
    return float(np.mean(block, dtype=np.float64))

def block_variance(block):
    return float(np.var(block, dtype=np.float64))

def block_min(block):
    return float(np.min(block))

def block_max(block):
    return float(np.max(block))

def block_autocorrelation(block):
    # full autocorrelation, take zero-lag
    ac = np.correlate(block, block, mode='full')
    return float(ac[len(block) - 1])

def block_fft(block):
    # block is already float64
    return np.abs(np.fft.rfft(block.astype(np.float64)))

def spectral_energy(fft_vals):
    return float(np.sum(fft_vals ** 2, dtype=np.float64))

def spectral_centroid(fft_vals, sample_rate):
    if len(fft_vals) == 0:
        return 0.0
    freqs = np.linspace(0.0, sample_rate / 2.0, len(fft_vals), dtype=np.float64)
    num = np.sum(freqs * fft_vals, dtype=np.float64)
    den = np.sum(fft_vals, dtype=np.float64)
    return float(num / den) if den != 0.0 else 0.0

# -----------------------------
# RECORD ONE TEMPORAL SAMPLE
# -----------------------------
def record_temporal_sample():
    init_dataset()

    block = capture_block()

    ts = datetime.now(UTC)
    ts_iso = ts.isoformat()
    ts_unix = ts.timestamp()

    mean_val = block_mean(block)
    var_val = block_variance(block)
    min_val = block_min(block)
    max_val = block_max(block)
    auto_val = block_autocorrelation(block)

    fft_vals = block_fft(block)
    energy_val = spectral_energy(fft_vals)
    centroid_val = spectral_centroid(fft_vals, SAMPLE_RATE)

    time_since_start = time.time() - PROCESS_START_UNIX
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage(WORKING_DIR).percent
    system_boot_time_unix = psutil.boot_time()

    # write to file
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts_iso,
            ts_unix,
            mean_val,
            var_val,
            min_val,
            max_val,
            auto_val,
            energy_val,
            centroid_val,
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

    # print to screen (raw temporal dynamics, 64-bit)
    print(
        f"[{ts_iso}] "
        f"mean={mean_val:.6f} var={var_val:.6f} "
        f"min={min_val:.6f} max={max_val:.6f} "
        f"auto={auto_val:.3e} energy={energy_val:.3e} centroid={centroid_val:.2f}"
    )

# -----------------------------
# MAIN LOOP
# -----------------------------
print("Starting RAW TEMPORAL PROBE (64-BIT)...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Device index:", sd.default.device)
print("Sample rate:", SAMPLE_RATE)
print("Chunk size:", CHUNK_SIZE)
print("Output file:", OUTPUT_FILE)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds\n")

while True:
    record_temporal_sample()
    time.sleep(SAMPLING_INTERVAL_SECONDS)
