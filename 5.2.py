from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Put the stereo input WAV beside this Python script.
script_dir = Path(__file__).resolve().parent
input_path = script_dir / "Am_AcusticPlug28_1.wav"

sample_rate, original = wavfile.read(input_path)
if original.ndim != 2 or original.shape[1] != 2:
    raise ValueError("This script expects a stereo WAV file with two channels.")

# Convert integer PCM samples to floating point.
if np.issubdtype(original.dtype, np.integer):
    info = np.iinfo(original.dtype)
    scale = max(abs(info.min), info.max)
    audio = original.astype(np.float32) / scale
else:
    audio = original.astype(np.float32)

num_samples, num_channels = audio.shape

# Echo settings.
alpha = 0.5
D0_seconds = 0.020        # Nominal delay: 20 ms
delta_D_seconds = 0.005   # Delay varies by +/- 5 ms
fm = 1.0                  # Modulation frequency: 1 Hz

D0 = round(D0_seconds * sample_rate)
delta_D = round(delta_D_seconds * sample_rate)

n = np.arange(num_samples)

# Fixed-delay echo, for comparison.
fixed = audio.copy()
if D0 < num_samples:
    fixed[D0:] += alpha * audio[:-D0]

# Time-varying delay: D(n) = D0 + delta_D * sin(2*pi*fm*n/fs).
variable_delays = np.rint(
    D0 + delta_D * np.sin(2 * np.pi * fm * n / sample_rate)
).astype(int)

time_varying = audio.copy()
source_indices = n - variable_delays
valid = source_indices >= 0

# Apply the variable delay independently to the left and right channels.
for channel in range(num_channels):
    time_varying[valid, channel] += (
        alpha * audio[source_indices[valid], channel]
    )


# Save the stereo output as a floating-point WAV file.
def save_float_wav(path, signal):
    peak = np.max(np.abs(signal))
    if peak > 1:
        signal = signal / peak
    wavfile.write(path, sample_rate, signal.astype(np.float32))


fixed_path = script_dir / "fixed_delay_echo_5_2.wav"
variable_path = script_dir / "time_varying_echo_5_2.wav"

save_float_wav(fixed_path, fixed)
save_float_wav(variable_path, time_varying)

print(f"Saved fixed-delay echo: {fixed_path}")
print(f"Saved time-varying echo: {variable_path}")

# Plot the first 0.1 seconds for the left and right channels.
view_samples = min(num_samples, int(0.1 * sample_rate))
time = np.arange(view_samples) / sample_rate

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for channel, axis in enumerate(axes):
    channel_name = "Left" if channel == 0 else "Right"
    axis.plot(time, audio[:view_samples, channel], label="Original", alpha=0.8)
    axis.plot(time, fixed[:view_samples, channel], label="Fixed-delay echo", alpha=0.8)
    axis.plot(time, time_varying[:view_samples, channel], label="Time-varying echo", alpha=0.8)
    axis.set_ylabel("Amplitude")
    axis.set_title(f"{channel_name} channel")
    axis.legend()
    axis.grid(True)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Waveform comparison (first 0.1 seconds)")
fig.tight_layout()


def fft(signal):
    """Compute the FFT using the Cooley-Tukey radix-2 algorithm.

    The input is zero-padded to the next power of two.
    """
    values = [complex(value) for value in signal]

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


# Compare spectra from the first five seconds for both channels.
segment_samples = min(num_samples, int(5 * sample_rate))


def magnitude_spectrum(signal, channel):
    segment = signal[:segment_samples, channel]
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


fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for channel, axis in enumerate(axes):
    channel_name = "Left" if channel == 0 else "Right"
    f_original, mag_original = magnitude_spectrum(audio, channel)
    f_fixed, mag_fixed = magnitude_spectrum(fixed, channel)
    f_variable, mag_variable = magnitude_spectrum(time_varying, channel)

    axis.plot(f_original, mag_original, label="Original", alpha=0.8)
    axis.plot(f_fixed, mag_fixed, label="Fixed-delay echo", alpha=0.8)
    axis.plot(f_variable, mag_variable, label="Time-varying echo", alpha=0.8)
    axis.set_ylabel("Magnitude (dB)")
    axis.set_title(f"{channel_name} channel")
    axis.legend()
    axis.grid(True)

axes[-1].set_xlabel("Frequency (Hz)")
fig.suptitle("Magnitude spectrum comparison (first 5 seconds)")
fig.tight_layout()
plt.show()
