import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from scipy.signal import savgol_filter

# Read the WAV file
sample_rate, data = wavfile.read('audio_with_noise_2.wav')
n_grad_freq=2; n_grad_time=4; n_fft=2048; win_length=2048; hop_length=512; n_std_thresh=1.5; prop_decrease=0.6

data = data / 32768
# If stereo, take only one channel
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Perform rFFT
fft_spectrum = np.fft.rfft(data)
freq = np.fft.rfftfreq(len(data), 1 / sample_rate)

audio_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

# Get the amplitude
amplitude = np.abs(fft_spectrum)

# Convert amplitude to dB
amplitude_db = 10 * np.log10(amplitude)

librosa_db = librosa.amplitude_to_db(amplitude)
librosa_db[librosa_db == -np.inf] = 0

# Apply a mean filter (moving average) to the decibel values
window_length = 205    # Length of the filter window (must be odd)
polyorder = 3       # Order of the polynomial used to fit the samples
smoothed_amplitude_db = savgol_filter(librosa_db, window_length, polyorder)
new_smoothed_amplitude_db = np.array([smoothed_amplitude_db[np.where(abs(freq - audio_freqs[i]) < 0.5)[0][0]] for i in range(len(audio_freqs))])
# Create subplots
fig, axs = plt.subplots(5, 1, figsize=(14, 10))

audio_stft = librosa.stft(data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
audio_stft_db = librosa.amplitude_to_db(np.abs(audio_stft))
audio_stft_mean = np.mean(audio_stft_db, axis=1)
axs[0].plot(audio_freqs, audio_stft_mean)
axs[0].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel('Amplitude [dB]')
axs[0].set_title('Amplitude Spectrum STFT Mean')
axs[0].grid()

axs[1].plot(freq, amplitude_db)
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Amplitude [dB]')
axs[1].set_title('Amplitude Spectrum FFT')
axs[1].grid()

axs[2].plot(freq, smoothed_amplitude_db)
axs[2].set_xlabel('Frequency [Hz]')
axs[2].set_ylabel('Amplitude [dB]')
axs[2].set_title('Smoothed Amplitude Spectrum FFT')
axs[2].grid()

axs[3].plot(audio_freqs, new_smoothed_amplitude_db)
axs[3].set_xlabel('Frequency [Hz]')
axs[3].set_ylabel('Amplitude [dB]')
axs[3].set_title('Reduced Smoothed Amplitude Spectrum FFT')
axs[3].grid()

fft_spectrum = np.fft.rfft(data, n=n_fft)
amplitude = np.abs(fft_spectrum)
librosa_db = librosa.amplitude_to_db(amplitude)
librosa_db[librosa_db == -np.inf] = 0

# Apply a mean filter (moving average) to the decibel values
window_length = 205    # Length of the filter window (must be odd)
polyorder = 3       # Order of the polynomial used to fit the samples
smoothed_amplitude_db = savgol_filter(librosa_db, window_length, polyorder)

axs[4].plot(audio_freqs, smoothed_amplitude_db)
axs[4].set_xlabel('Frequency [Hz]')
axs[4].set_ylabel('Amplitude [dB]')
axs[4].set_title('Cropped Smoothed Amplitude Spectrum FFT')
axs[4].grid()

plt.tight_layout()
plt.savefig(f'hehehehehe.png')

