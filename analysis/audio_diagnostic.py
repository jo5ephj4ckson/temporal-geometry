import sounddevice as sd
import numpy as np
import time

# Set your working device and sample rate
sd.default.device = 12      # or 18 if 12 is silent
SAMPLE_RATE = 48000
BLOCK_SIZE = 2048

print("Starting long‑running diagnostic...")
print("Press CTRL+C to stop.\n")

while True:
    try:
        block = sd.rec(BLOCK_SIZE, samplerate=SAMPLE_RATE,
                       channels=1, dtype='float64')
        sd.wait()

        block = block.flatten()
        var_val = np.var(block)
        mean_val = np.mean(block)
        min_val = np.min(block)
        max_val = np.max(block)

        print(f"var={var_val:.8f}  mean={mean_val:.6f}  min={min_val:.6f}  max={max_val:.6f}")

        time.sleep(1)

    except KeyboardInterrupt:
        print("\nDiagnostic stopped.")
        break
