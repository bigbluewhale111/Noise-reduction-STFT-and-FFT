import time
import denoise
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and denoise audio files.")
    parser.add_argument("audio_with_noise", type=str, help="Path to the noisy audio file.")
    parser.add_argument("noise_profile", type=str, help="Path to the noise profile audio file.")
    parser.add_argument("recover_name", type=str, help="Recover files name")
    parser.add_argument("--n_grad_freq", type=int, default=2, help="Number of frequency gradient steps.")
    parser.add_argument("--n_grad_time", type=int, default=4, help="Number of time gradient steps.")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size.")
    parser.add_argument("--win_length", type=int, default=2048, help="Window length for FFT.")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length for FFT.")
    parser.add_argument("--n_std_thresh", type=float, default=1.5, help="Threshold in terms of standard deviations.")
    parser.add_argument("--prop_decrease", type=float, default=0.6, help="Proportion to decrease noise.")
    parser.add_argument("--visual", action="store_true", help="Display visual plots.")
    args = parser.parse_args()
    time_1 = time.time()
    denoise.denoise(args.audio_with_noise,args.noise_profile, args.recover_name, n_grad_freq=args.n_grad_freq, n_grad_time=args.n_grad_time, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_std_thresh=args.n_std_thresh, prop_decrease=args.prop_decrease, visual=args.visual)
    time_2 = time.time()
    denoise.denoise_v2(args.audio_with_noise,args.noise_profile, args.recover_name, n_grad_freq=args.n_grad_freq, n_grad_time=args.n_grad_time, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_std_thresh=args.n_std_thresh, prop_decrease=args.prop_decrease, visual=args.visual)
    time_3 = time.time()
    print(f"STFT denoise finish in {time_2 - time_1}\nFFT denoise finish in {time_3 - time_2}")