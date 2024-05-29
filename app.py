import time
import denoise

time_1 = time.time()
denoise.denoise('audio_with_noise.wav','noise.wav', 'recover')
time_2 = time.time()
denoise.denoise_v2('audio_with_noise.wav','noise.wav', 'recover_v2')
time_3 = time.time()
print(f"STFT denoise finish in {time_2 - time_1}\nFFT denoise finish in {time_3 - time_2}")