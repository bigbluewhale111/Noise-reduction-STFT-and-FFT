import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import librosa
import pywt
import time

# Define the sample rate and the time duration for each frequency segment
sample_rate = 5000  # samples per second
t1 = np.arange(0, 5, 1/sample_rate)
t2 = np.arange(5, 10, 1/sample_rate)
t3 = np.arange(10, 15, 1/sample_rate)
t4 = np.arange(15, 20, 1/sample_rate)

# Define the frequencies
f1 = 100  # 100 Hz
f2 = 250  # 250 Hz
f3 = 500  # 500 Hz
f4 = 1000 # 1000 Hz

# Create the signals for each frequency
signal1 = np.sin(2 * np.pi * f1 * t1)
signal2 = np.sin(2 * np.pi * f2 * t2)
signal3 = np.sin(2 * np.pi * f3 * t3)
signal4 = np.sin(2 * np.pi * f4 * t4)

# Concatenate the signals to form the complete signal
complete_signal = np.concatenate((signal1, signal2, signal3, signal4))

# Create a time array for the complete signal
t = np.arange(0, 20, 1/sample_rate)

# Apply the real FFT
_t1 = time.time()
fft_result = rfft(complete_signal)
_t2 = time.time()

print(f"FFT calculated in {_t2 - _t1}")
# Generate the corresponding frequency bins
freqs = rfftfreq(len(complete_signal), 1/sample_rate)

# Calculate STFT of audio using librosa
_t1 = time.time()
audio_stft = librosa.stft(complete_signal, n_fft=1024, win_length=1024, hop_length=256)
_t2 = time.time()
print(f"STFT calcuclated in {_t2 - _t1}")
audio_stft_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max)
audio_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)

# CWT
_t1 = time.time()
scales = pywt.frequency2scale('cmor1.5-1.0', audio_freqs[1:]/sample_rate)
coeffs, _freqs = pywt.cwt(complete_signal, scales, 'cmor1.5-1.0', sampling_period=1/sample_rate)
_t2 = time.time()
print(f"CWT calcuclated in {_t2 - _t1}")
cwt_db = np.abs(coeffs[:-1, :-1])
time = np.append(t1, [t2, t3, t4])

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 10))

# Plot the magnitude of the FFT result
axs[0].plot(freqs, np.abs(fft_result))
axs[0].set_title('Magnitude Spectrum of the Signal using FFT analysis')
axs[0].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel('Magnitude')
axs[0].grid()

# Plot the magnitude of the STFT result
img = librosa.display.specshow(audio_stft_db, sr=sample_rate, hop_length=256, x_axis='time', y_axis='hz', ax=axs[1], cmap=plt.cm.Blues)
fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_title('Magnitude Spectrum of the Signal using STFT analysis')
axs[1].grid()

pcm = axs[2].pcolormesh(time, _freqs, cwt_db, cmap=plt.cm.Blues)
fig.colorbar(pcm, ax=axs[2], format='%+2.0f dB')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_title('Magnitude Spectrum of the Signal using CWT analysis')
axs[2].grid()


# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.savefig('fft_stft_demonstration.png')
