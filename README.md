# Noise-reduction-STFT-and-FFT
## Installation
```sh
pip install -r requirements.txt
```
## Running
```sh
python app.py "AUDIO_WITH_NOISE.wav" "NOISE_PROFILE.wav"
```
`-h` for help
## Files
| Filename   | Description                                                  |
|------------|--------------------------------------------------------------|
| `app.py`   | Running code. Entry point of the application.                |
| `denoise.py` | Our denoise code. Contains functions for signal denoising. |
| `plot.py`  | Plot the visual comparison of FFT, STFT and CWT.             |
| `plot_2.py`| Plot the visual comparison of FFT method and STFT method.    |
