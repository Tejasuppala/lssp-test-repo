LSSP MIDTERM / FINAL PROJECT
Audio Signal Processing Experiments
===================================

OVERVIEW
--------
This project contains two Jupyter Notebook experiments using a stereo WAV recording:

  5.1.ipynb  Room reverberation modeled as a linear time-invariant (LTI) system.
  5.2.ipynb  Musical echo effects modeled with a time-varying delay.

The input recording is Am_AcusticPlug28_1.wav. The audio outputs and plots included
with the project are organized in the audio outputs and plots folders.

EXPERIMENT 5.1 - ROOM REVERBERATION
-----------------------------------
The recording is convolved with an impulse response consisting of a direct sound
and delayed reflections:

  h[n] = delta[n] + sum(a_i * delta[n - d_i])

The notebook implements convolution manually and processes the left and right
channels independently. It compares:

  - Five reflections spaced 0.10 seconds apart, with amplitudes 2^(-i) and 3^(-i).
  - Three, five, and eight reflections at 0.10-second spacing.
  - Five reflections at 0.05-, 0.10-, and 0.20-second spacing.

The count and spacing comparisons use the 3^(-i) amplitude rule. The notebook
plots the impulse responses and compares original and processed magnitude spectra
for both channels, using the first five seconds of audio. Its FFT is a recursive
radix-2 Cooley-Tukey implementation.

The five-reflection, 0.10-second case appears in both the count and spacing
comparisons. These settings are identical, so the separately named WAV files
in audio outputs contain the same audio.

EXPERIMENT 5.2 - TIME-VARYING DELAY
-----------------------------------
This notebook compares the original recording with a fixed-delay echo and a
periodically changing echo delay:

  y[n] = x[n] + alpha * x[n - D(n)]
  D(n) = D0 + delta_D * sin(2*pi*fm*n/fs)

Default settings are alpha = 0.5, D0 = 20 ms, delta_D = 5 ms, and fm = 1 Hz.
The notebook compares stereo waveforms over the first 0.1 seconds and magnitude
spectra over the first five seconds. It also uses a recursive manual FFT.

PROJECT CONTENTS
-----------------
  source code/
    5.1.ipynb                  Room reverberation notebook
    5.2.ipynb                  Time-varying delay notebook
    Am_AcusticPlug28_1.wav     Stereo input recording
  audio outputs/               Generated WAV files from both experiments
  plots/                       Saved impulse-response, waveform, and spectrum plots

REQUIREMENTS
------------
Use Python with Jupyter Notebook, NumPy, SciPy, and Matplotlib. SciPy's wavfile
module is used for WAV input/output; the convolution and FFT algorithms are
implemented in the notebooks.

RUNNING THE NOTEBOOKS
---------------------
1. Install the required packages if they are not already installed:

     python -m pip install numpy scipy matplotlib jupyter

2. Open the notebook from the source code folder and select a Python kernel.
3. Keep the notebook working directory set to source code so the input WAV can
   be found, then run the cells from top to bottom.

Note: The 5.1 notebook's save cells use the variable script_dir. If it is undefined
in a fresh kernel, set script_dir = Path.cwd() in the first setup cell before
running the notebook.

OUTPUTS
-------
The audio outputs folder contains the reverberated recordings and echo recordings.
The plots folder contains PNG figures for the impulse responses, waveforms, and
magnitude spectra. Re-running cells may create new outputs depending on the
notebook's working directory and the output paths in the cells.