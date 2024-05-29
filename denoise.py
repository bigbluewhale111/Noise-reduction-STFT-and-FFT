from scipy.io import wavfile
from scipy.fft import rfft, ifft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.ndimage
import math
import time

def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png')

def denoise_v2(input_audio, noise_profile, recover_file, n_grad_freq=2, n_grad_time=4, n_fft=2048, win_length=2048, hop_length=512, n_std_thresh=1.5, prop_decrease=0.6, visual=False):
    noise_sample_rate, noise_data = wavfile.read(noise_profile)
    sample_rate, data = wavfile.read(input_audio)

    assert noise_sample_rate == sample_rate, "Audio and noise sample rates are not same"

    noise_data = noise_data / 32768
    if len(noise_data.shape) > 1:
        noise_data = noise_data.mean(axis=1)
    noise_fft = rfft(noise_data)
    freq = np.fft.rfftfreq(len(noise_data), 1 / sample_rate)
    noise_fft_db = librosa.amplitude_to_db(np.abs(noise_fft), ref=1.0, amin=1e-20, top_db=80.0)
    audio_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    window_length = 205    # Length of the filter window (must be odd)
    polyorder = 3       # Order of the polynomial used to fit the samples
    smoothed_amplitude_db = scipy.signal.savgol_filter(noise_fft_db, window_length, polyorder)
    noise_thresh = np.array([smoothed_amplitude_db[np.where(abs(freq - audio_freqs[i]) < 0.5)[0][0]] for i in range(audio_freqs.shape[0])]) - np.ones(audio_freqs.shape[0])*5.88*n_std_thresh
    
    data = data / 32768
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    audio_stft = librosa.stft(data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    audio_stft_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=1.0, amin=1e-20, top_db=80.0)
    if visual:
        plot_spectrogram(audio_stft_db, 'original_v2')
    mask_gain_dB = np.min(audio_stft_db)

    smoothing_filter = np.outer(
        np.concatenate(
        [
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # print(f"smoothing_filter.shape: {smoothing_filter.shape}")
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(noise_thresh)]),
        np.shape(audio_stft_db)[1],
        axis=0,
    ).T
    if visual:
        plot_spectrogram(db_thresh, "db_thresh_v2")
    # print(f"db_thresh.shape: {db_thresh.shape}")
    sig_mask = (audio_stft_db < db_thresh)
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if visual:
        plot_spectrogram(sig_mask, "mask_v2")

    sig_real_masked = (
        audio_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(audio_stft_db) * (1 - sig_mask)
    sig_masked_amp = (librosa.db_to_amplitude(sig_real_masked, ref=1.0) * np.sign(audio_stft)) + (
        1j * sig_imag_masked
    )
    if visual:
        plot_spectrogram(sig_real_masked, 'sig_masked_v2')
    
    recover_audio_sig = librosa.istft(sig_masked_amp, win_length=win_length, hop_length=hop_length)
    if visual:
        recover_stft = librosa.amplitude_to_db(np.abs(librosa.stft(recover_audio_sig, n_fft=n_fft, win_length=win_length, hop_length=hop_length)), ref=1.0, amin=1e-20, top_db=80.0)
        plot_spectrogram(recover_stft, recover_file)
    wavfile.write(recover_file+'.wav', sample_rate, (recover_audio_sig*32768).astype(np.int16))

def denoise(input_audio, noise_profile, recover_file, n_grad_freq=2, n_grad_time=4, n_fft=2048, win_length=2048, hop_length=512, n_std_thresh=1.5, prop_decrease=0.6, visual=False):
    noise_sample_rate, noise_data = wavfile.read(noise_profile)
    sample_rate, data = wavfile.read(input_audio)
    assert noise_sample_rate == sample_rate, "Audio and noise sample rates are not same"

    noise_data = noise_data / 32768
    if len(noise_data.shape) > 1:
        noise_data = noise_data.mean(axis=1)
    noise_stft = librosa.stft(noise_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    noise_stft_db = librosa.amplitude_to_db(np.abs(noise_stft), ref=1.0, amin=1e-20, top_db=80.0)
    noise_stft_mean = np.mean(noise_stft_db, axis=1)
    noise_stft_std = np.std(noise_stft_db, axis=1)
    noise_thresh = noise_stft_mean + noise_stft_std * n_std_thresh

    data = data / 32768
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    audio_stft = librosa.stft(data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    audio_stft_db = librosa.amplitude_to_db(np.abs(audio_stft), ref=1.0, amin=1e-20, top_db=80.0)
    if visual:
        plot_spectrogram(audio_stft_db, 'original')
    mask_gain_dB = np.min(audio_stft_db)

    smoothing_filter = np.outer(
        np.concatenate(
        [
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # print(f"smoothing_filter.shape: {smoothing_filter.shape}")
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(noise_stft_mean)]),
        np.shape(audio_stft_db)[1],
        axis=0,
    ).T
    if visual:
        plot_spectrogram(db_thresh, "db_thresh")
    # print(f"db_thresh.shape: {db_thresh.shape}")
    sig_mask = (audio_stft_db < db_thresh)
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    # print(f"sig_mask.shape: {sig_mask.shape}")
    if visual:
        plot_spectrogram(sig_mask, "mask")

    sig_real_masked = (
        audio_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(audio_stft_db) * (1 - sig_mask)
    sig_masked_amp = (librosa.db_to_amplitude(sig_real_masked, ref=1.0) * np.sign(audio_stft)) + (
        1j * sig_imag_masked
    )
    if visual:
        plot_spectrogram(sig_real_masked, 'sig_masked')
    
    recover_audio_sig = librosa.istft(sig_masked_amp, win_length=win_length, hop_length=hop_length)
    if visual:
        recover_stft = librosa.amplitude_to_db(np.abs(librosa.stft(recover_audio_sig, n_fft=n_fft, win_length=win_length, hop_length=hop_length)), ref=1.0, amin=1e-20, top_db=80.0)
        plot_spectrogram(recover_stft, recover_file)
    wavfile.write(recover_file+'.wav', sample_rate, (recover_audio_sig*32768).astype(np.int16))

