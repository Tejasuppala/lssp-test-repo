from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Put the stereo input WAV beside this Python script.
script_dir = Path(__file__).resolve().parent
input_path = script_dir / "Am_AcusticPlug28_1.wav"
output_2_path = script_dir / "reverberated_2power.wav"
output_3_path = script_dir / "reverberated_3power.wav"

# Convert PCM integers to floating point.
if np.issubdtype(original.dtype, np.integer):
    info = np.iinfo(original.dtype)
    scale = max(abs(info.min), info.max)
    audio = original.astype(np.float32) / scale
else:
    audio = original.astype(np.float32)

# Five reflections at 0.10-second intervals.
delays_seconds = [0.10, 0.20, 0.30, 0.40, 0.50]
delay_samples = [round(delay * sample_rate) for delay in delays_seconds]

# Compare two amplitude patterns: 2^-i and 3^-i.
amplitudes_2 = [2 ** (-i) for i in range(1, len(delays_seconds) + 1)]
amplitudes_3 = [3 ** (-i) for i in range(1, len(delays_seconds) + 1)]


def make_impulse_response(amplitudes):
    impulse_response = np.zeros(delay_samples[-1] + 1, dtype=np.float32)
    impulse_response[0] = 1.0
    for delay, amplitude in zip(delay_samples, amplitudes):
        impulse_response[delay] += amplitude
    return impulse_response


impulse_response_2 = make_impulse_response(amplitudes_2)
impulse_response_3 = make_impulse_response(amplitudes_3)


def manual_convolve(signal, impulse_response):
    """Manually convolve one channel, skipping zero-valued impulse taps."""
    output = np.zeros(
        len(signal) + len(impulse_response) - 1,
        dtype=np.float32
    )

    for k, tap in enumerate(impulse_response):
        if tap != 0:
            for n in range(len(signal)):
                output[n + k] += signal[n] * tap

    return output


def process_stereo(impulse_response):
    return np.column_stack([
        manual_convolve(audio[:, channel], impulse_response)
        for channel in range(2)
    ])


processed_2 = process_stereo(impulse_response_2)
processed_3 = process_stereo(impulse_response_3)

# Apply one shared gain if needed, preserving the relative level of both versions.
peak = max(
    float(np.max(np.abs(audio))),
    float(np.max(np.abs(processed_2))),
    float(np.max(np.abs(processed_3)))
)
if peak > 1:
    audio = audio / peak
    processed_2 = processed_2 / peak
    processed_3 = processed_3 / peak

wavfile.write(output_2_path, sample_rate, processed_2.astype(np.float32))
wavfile.write(output_3_path, sample_rate, processed_3.astype(np.float32))
print(f"Saved 2-power reverberation: {output_2_path}")
print(f"Saved 3-power reverberation: {output_3_path}")

# Plot the two impulse responses.
time_2 = np.arange(len(impulse_response_2)) / sample_rate
time_3 = np.arange(len(impulse_response_3)) / sample_rate

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].stem(time_2, impulse_response_2)
axes[0].set_title("Reflection amplitudes: powers of 2 (2^-i)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[1].stem(time_3, impulse_response_3)
axes[1].set_title("Reflection amplitudes: powers of 3 (3^-i)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)
fig.tight_layout()


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


# Compare the original and both processed spectra for left and right channels.
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for channel, axis in enumerate(axes):
    f_original, mag_original = magnitude_spectrum(audio[:, channel])
    f_2, mag_2 = magnitude_spectrum(processed_2[:, channel])
    f_3, mag_3 = magnitude_spectrum(processed_3[:, channel])

    channel_name = "Left" if channel == 0 else "Right"
    axis.plot(f_original, mag_original, label="Original", alpha=0.8)
    axis.plot(f_2, mag_2, label="2-power reflections", alpha=0.8)
    axis.plot(f_3, mag_3, label="3-power reflections", alpha=0.8)
    axis.set_ylabel("Magnitude (dB)")
    axis.set_title(f"{channel_name} channel")
    axis.legend()
    axis.grid(True)

axes[-1].set_xlabel("Frequency (Hz)")
fig.suptitle("Original and reverberated magnitude spectra")
fig.tight_layout()
plt.show()
