import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
CHANNELS = 1
NUM_BLOCKS = 200   # number of entropy blocks to analyze

# -----------------------------
# 1. Capture raw tunneling entropy blocks
# -----------------------------
def get_entropy_block():
    data = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return data.flatten().astype(np.float64)

tunneling_blocks = []
for _ in range(NUM_BLOCKS):
    tunneling_blocks.append(get_entropy_block())

# -----------------------------
# 2. Generate synthetic noise blocks
# -----------------------------
synthetic_blocks = []
for _ in range(NUM_BLOCKS):
    synthetic_blocks.append(np.random.normal(0, 1, CHUNK_SIZE))

# -----------------------------
# 3. Compute statistics
# -----------------------------
def compute_stats(blocks):
    means = [np.mean(b) for b in blocks]
    vars_ = [np.var(b) for b in blocks]
    autocorrs = [np.correlate(b, b, mode='full')[CHUNK_SIZE-1] for b in blocks]
    return means, vars_, autocorrs

tun_mean, tun_var, tun_auto = compute_stats(tunneling_blocks)
syn_mean, syn_var, syn_auto = compute_stats(synthetic_blocks)

# -----------------------------
# 4. FFT spectral density
# -----------------------------
def compute_fft(blocks):
    spectra = []
    for b in blocks:
        fft_vals = np.abs(np.fft.rfft(b))
        spectra.append(fft_vals)
    return np.mean(spectra, axis=0)

tun_fft = compute_fft(tunneling_blocks)
syn_fft = compute_fft(synthetic_blocks)

# -----------------------------
# 5. Print summary
# -----------------------------
print("TUNNELING ENTROPY:")
print("  Mean amplitude (avg):", np.mean(tun_mean))
print("  Variance (avg):", np.mean(tun_var))
print("  Autocorrelation (avg):", np.mean(tun_auto))

print("\nSYNTHETIC NOISE:")
print("  Mean amplitude (avg):", np.mean(syn_mean))
print("  Variance (avg):", np.mean(syn_var))
print("  Autocorrelation (avg):", np.mean(syn_auto))

# -----------------------------
# 6. Plot comparison
# -----------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Mean Amplitude")
plt.plot(tun_mean, label="Tunneling")
plt.plot(syn_mean, label="Synthetic")
plt.legend()

plt.subplot(2,2,2)
plt.title("Variance")
plt.plot(tun_var, label="Tunneling")
plt.plot(syn_var, label="Synthetic")
plt.legend()

plt.subplot(2,2,3)
plt.title("Autocorrelation")
plt.plot(tun_auto, label="Tunneling")
plt.plot(syn_auto, label="Synthetic")
plt.legend()

plt.subplot(2,2,4)
plt.title("FFT Spectral Density")
plt.plot(tun_fft, label="Tunneling")
plt.plot(syn_fft, label="Synthetic")
plt.legend()

plt.tight_layout()
plt.show()
