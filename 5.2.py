from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


# Find the input WAV beside this script.
script_dir = Path(__file__).resolve().parent
input_path = script_dir / "Am_AcusticPlug28_1.wav"

sample_rate, original = wavfile.read(input_path)

# Convert integer PCM samples to floating point.
if np.issubdtype(original.dtype, np.integer):
    info = np.iinfo(original.dtype)
    scale = max(abs(info.min), info.max)
    audio = original.astype(np.float32) / scale
else:
    audio = original.astype(np.float32)

# Make audio consistently 2-D so mono and stereo use the same code.
if audio.ndim == 1:
    channels = audio[:, np.newaxis]
else:
    channels = audio

num_samples, num_channels = channels.shape


# Echo settings.
alpha = 0.5
D0_seconds = 0.020        # Nominal delay: 20 ms
delta_D_seconds = 0.005   # Delay varies by +/- 5 ms
fm = 1.0                  # Modulation frequency: 1 Hz

D0 = round(D0_seconds * sample_rate)
delta_D = round(delta_D_seconds * sample_rate)

n = np.arange(num_samples)


# Fixed-delay echo.
fixed = channels.copy()

if D0 < num_samples:
    fixed[D0:] += alpha * channels[:-D0]


# Time-varying delay:
# D(n) = D0 + delta_D * sin(2*pi*fm*n/sample_rate)
variable_delays = np.rint(
    D0 + delta_D * np.sin(2 * np.pi * fm * n / sample_rate)
).astype(int)

time_varying = channels.copy()
source_indices = n - variable_delays
valid = source_indices >= 0

for channel in range(num_channels):
    time_varying[valid, channel] += (
        alpha * channels[source_indices[valid], channel]
    )


# Save both outputs as floating-point WAV files.
def save_float_wav(path, signal):
    # Reduce the signal only if it would exceed the normal audio range.
    peak = np.max(np.abs(signal))
    if peak > 1:
        signal = signal / peak

    # Write mono audio as a 1-D array.
    if signal.shape[1] == 1:
        signal = signal[:, 0]

    wavfile.write(path, sample_rate, signal.astype(np.float32))


fixed_path = script_dir / "fixed_delay_echo_5_2.wav"
variable_path = script_dir / "time_varying_echo_5_2.wav"

save_float_wav(fixed_path, fixed)
save_float_wav(variable_path, time_varying)

print(f"Saved fixed-delay echo: {fixed_path}")
print(f"Saved time-varying echo: {variable_path}")


# Plot the first 0.1 seconds of channel 1.
view_samples = min(num_samples, int(0.1 * sample_rate))
time = np.arange(view_samples) / sample_rate

plt.figure(figsize=(10, 4))
plt.plot(time, channels[:view_samples, 0], label="Original", alpha=0.8)
plt.plot(time, fixed[:view_samples, 0], label="Fixed-delay echo", alpha=0.8)
plt.plot(
    time,
    time_varying[:view_samples, 0],
    label="Time-varying echo",
    alpha=0.8
)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform comparison (first 0.1 seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()


def fft(signal):
    """Compute the FFT using the Cooley–Tukey algorithm.

    The input is zero-padded to the next power of two.
    """
    values = [complex(value) for value in signal]

    # Pad the input length to a power of two.
    fft_length = 1
    while fft_length < len(values):
        fft_length *= 2
    values.extend([0j] * (fft_length - len(values)))

    # Reorder values by reversing their binary indices.
    j = 0
    for i in range(1, fft_length):
        bit = fft_length >> 1

        while j & bit:
            j ^= bit
            bit >>= 1

        j ^= bit

        if i < j:
            values[i], values[j] = values[j], values[i]

    # Calculate the FFT in stages.
    size = 2
    while size <= fft_length:
        half_size = size // 2
        angle = -2 * math.pi / size
        step = complex(math.cos(angle), math.sin(angle))

        for start in range(0, fft_length, size):
            twiddle = 1 + 0j

            for offset in range(half_size):
                even = values[start + offset]
                odd = twiddle * values[start + offset + half_size]

                values[start + offset] = even + odd
                values[start + offset + half_size] = even - odd
                twiddle *= step

        size *= 2

    return values


# Compare spectra using the first five seconds to keep plotting manageable.
segment_samples = min(num_samples, int(5 * sample_rate))


def magnitude_spectrum(signal):
    # Use channel 1 and the selected short segment.
    segment = signal[:segment_samples, 0]

    spectrum = fft(segment)
    fft_length = len(spectrum)
    positive_length = fft_length // 2 + 1

    frequencies = [
        k * sample_rate / fft_length
        for k in range(positive_length)
    ]
    magnitude_db = [
        20 * math.log10(max(abs(spectrum[k]), 1e-10))
        for k in range(positive_length)
    ]

    return frequencies, magnitude_db


f_original, mag_original = magnitude_spectrum(channels)
f_fixed, mag_fixed = magnitude_spectrum(fixed)
f_variable, mag_variable = magnitude_spectrum(time_varying)

plt.figure(figsize=(10, 4))
plt.plot(f_original, mag_original, label="Original", alpha=0.8)
plt.plot(f_fixed, mag_fixed, label="Fixed-delay echo", alpha=0.8)
plt.plot(f_variable, mag_variable, label="Time-varying echo", alpha=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Magnitude spectrum comparison (first 5 seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()