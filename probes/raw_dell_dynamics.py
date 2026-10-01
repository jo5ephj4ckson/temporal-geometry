import csv
import os
import time
import platform
import psutil
import sounddevice as sd
import numpy as np
from datetime import datetime, UTC

# ============================================================
# DEVICE SELECTION (DELL WINDOWS 7 SAFE)
# ============================================================

def select_working_device():
    devices = sd.query_devices()
    candidates = []

    # Collect all devices that support 1 input channel
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) >= 1:
            candidates.append((idx, dev['name']))

    print("\nDetected input-capable devices:")
    for idx, name in candidates:
        print(f"  {idx}: {name}")

    # Priority order for Windows 7 stability:
    priority = ['DirectSound', 'MME', 'WASAPI', 'WDM-KS']

    for api in priority:
        for idx, name in candidates:
            if api.lower() in name.lower() or api.lower() in devices[idx]['hostapi'].__str__().lower():
                print(f"\nSelected device {idx}: {name} (API={api})")
                return idx

    # Fallback: first input-capable device
    print("\nFallback: using first input-capable device.")
    return candidates[0][0] if candidates else None


# ============================================================
# CONFIG
# ============================================================

DEVICE_INDEX = select_working_device()
if DEVICE_INDEX is None:
    raise RuntimeError("No usable audio input device found.")

sd.default.device = DEVICE_INDEX
sd.default.channels = 1
sd.default.samplerate = 48000

# WDM-KS exclusive mode fix
try:
    sd.default.blocksize = 0
except Exception:
    pass

AUDIO_DTYPE = 'int16'
CHUNK_SIZE = 2048
OUTPUT_FILE = "raw_temporal_dataset.csv"
SAMPLING_INTERVAL_SECONDS = 1

# ============================================================
# SYSTEM METADATA
# ============================================================

PROCESS_START_UNIX = time.time()
MACHINE_NAME = platform.node()
OS_INFO = f"{platform.system()} {platform.release()}"
PYTHON_VERSION = platform.python_version()
SCRIPT_NAME = os.path.basename(__file__) if "__file__" in globals() else "interactive"
WORKING_DIR = os.getcwd()

# ============================================================
# DATASET INIT
# ============================================================

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

# ============================================================
# TEMPORAL FEATURE FUNCTIONS
# ============================================================

def capture_block():
    try:
        block = sd.rec(CHUNK_SIZE,
                       samplerate=sd.default.samplerate,
                       channels=sd.default.channels,
                       dtype=AUDIO_DTYPE)
        sd.wait()
        return block.flatten()
    except Exception as e:
        print("ERROR capturing audio:", e)
        return np.zeros(CHUNK_SIZE, dtype=np.int16)

def block_mean(block): return float(np.mean(block))
def block_variance(block): return float(np.var(block))
def block_min(block): return int(np.min(block))
def block_max(block): return int(np.max(block))

def block_autocorrelation(block):
    ac = np.correlate(block, block, mode='full')
    return float(ac[len(block) - 1])

def block_fft(block):
    b = block.astype(np.float64)
    return np.abs(np.fft.rfft(b))

def spectral_energy(fft_vals):
    return float(np.sum(fft_vals ** 2))

def spectral_centroid(fft_vals, sample_rate):
    if len(fft_vals) == 0:
        return 0.0
    freqs = np.linspace(0, sample_rate / 2, len(fft_vals))
    num = np.sum(freqs * fft_vals)
    den = np.sum(fft_vals)
    return float(num / den) if den != 0 else 0.0

# ============================================================
# RECORD ONE SAMPLE
# ============================================================

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
    centroid_val = spectral_centroid(fft_vals, sd.default.samplerate)

    time_since_start = time.time() - PROCESS_START_UNIX
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage(WORKING_DIR).percent
    system_boot_time_unix = psutil.boot_time()

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

    print(
        f"[{ts_iso}] "
        f"mean={mean_val:.2f} var={var_val:.2f} "
        f"min={min_val} max={max_val} "
        f"auto={auto_val:.2e} energy={energy_val:.2e} centroid={centroid_val:.1f}"
    )

# ============================================================
# MAIN LOOP
# ============================================================

print("Starting RAW TEMPORAL PROBE...")
print("Machine:", MACHINE_NAME)
print("OS:", OS_INFO)
print("Python:", PYTHON_VERSION)
print("Selected device index:", sd.default.device)
print("Sample rate:", sd.default.samplerate)
print("Chunk size:", CHUNK_SIZE)
print("Output file:", OUTPUT_FILE)
print("Interval:", SAMPLING_INTERVAL_SECONDS, "seconds\n")

while True:
    record_temporal_sample()
    time.sleep(SAMPLING_INTERVAL_SECONDS)
