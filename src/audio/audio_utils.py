# audio_utils.py

import numpy as np
from scipy.signal import butter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sos

def apply_spectral_gate(signal, threshold=0.01):
    fft_sig = np.fft.rfft(signal)
    mag = np.abs(fft_sig)
    max_amp = np.max(mag)
    keep_mask = mag > (threshold * max_amp)
    fft_sig[~keep_mask] = 0.0
    out_sig = np.fft.irfft(fft_sig, n=len(signal))
    return out_sig

def preprocess_signal_for_vehicles(signal, fs=48000):
    # 1) DC removal
    signal = signal - np.mean(signal)

    # 2) Bandpass 50-3000 Hz
    sos = butter_bandpass(50.0, 3000.0, fs, order=4)
    filtered = sosfilt(sos, signal)

    # 3) Spectral gating
    gated = apply_spectral_gate(filtered, threshold=0.01)
    return gated
