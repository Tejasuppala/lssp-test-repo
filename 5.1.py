from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Put the input WAV beside this Python script.
script_dir = Path(__file__).resolve().parent
input_path = script_dir / "Am_AcusticPlug28_1.wav"
output_path = script_dir / "reverberated_3power.wav"

sample_rate, original = wavfile.read(input_path)

# Convert PCM integers to floating point.
if np.issubdtype(original.dtype, np.integer):
    info = np.iinfo(original.dtype)
    scale = max(abs(info.min), info.max)
    audio = original.astype(np.float32) / scale
else:
    audio = original.astype(np.float32)

# Five reflections at 0.10-second intervals.
delays_seconds = [0.10, 0.20, 0.30, 0.40, 0.50]

# Reflection amplitudes: 1/3, 1/9, 1/27, 1/81, 1/243.
amplitudes = [3 ** (-i) for i in range(1, len(delays_seconds) + 1)]

delay_samples = [round(delay * sample_rate) for delay in delays_seconds]
impulse_response = np.zeros(delay_samples[-1] + 1, dtype=np.float32)
impulse_response[0] = 1.0

for delay, amplitude in zip(delay_samples, amplitudes):
    impulse_response[delay] += amplitude


def manual_convolve(signal, impulse_response):
    """Convolve one 1-D signal with a 1-D impulse response."""
    output = np.zeros(
        len(signal) + len(impulse_response) - 1,
        dtype=np.float32
    )

    for n in range(len(signal)):
        for k in range(len(impulse_response)):
            output[n + k] += signal[n] * impulse_response[k]

    return output


# Apply the impulse response to mono or stereo audio.
if audio.ndim == 1:
    processed = manual_convolve(audio, impulse_response)
else:
    processed = np.column_stack([
        manual_convolve(audio[:, channel], impulse_response)
        for channel in range(audio.shape[1])
    ])

# Avoid clipping and save as floating-point WAV.
peak = np.max(np.abs(processed))
if peak > 1:
    processed = processed / peak

wavfile.write(output_path, sample_rate, processed.astype(np.float32))
print(f"Saved reverberant audio to: {output_path}")

# Plot the impulse response.
time = np.arange(len(impulse_response)) / sample_rate

plt.figure(figsize=(10, 3))
plt.stem(time, impulse_response)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Impulse response: reflection amplitudes decrease as powers of 1/3")
plt.grid(True)
plt.tight_layout()

# Compare magnitude spectra. For stereo, use the first channel.
original_mono = audio if audio.ndim == 1 else audio[:, 0]
processed_mono = processed if processed.ndim == 1 else processed[:, 0]


def fft(signal):
    """Compute an FFT with the radix-2 Cooley-Tukey algorithm.

    The signal is zero-padded to the next power of two.
    """
    values = [complex(value) for value in signal]

    fft_length = 1
    while fft_length < len(values):
        fft_length *= 2
    values.extend([0j] * (fft_length - len(values)))

    # Reorder samples by reversing their binary indices.
    j = 0
    for i in range(1, fft_length):
        bit = fft_length >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]

    # Combine pairs of smaller transforms into larger transforms.
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


def magnitude_spectrum(signal):
    spectrum = fft(signal)
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
f_original, mag_original = magnitude_spectrum(original_mono)
f_processed, mag_processed = magnitude_spectrum(processed_mono)

plt.figure(figsize=(10, 4))
plt.plot(f_original, mag_original, label="Original", alpha=0.8)
plt.plot(f_processed, mag_processed, label="Reverberated", alpha=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Original and reverberated magnitude spectra")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
