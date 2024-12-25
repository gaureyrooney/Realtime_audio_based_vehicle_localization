# gcc_methods.py

import numpy as np

def gcc(signal_1, signal_2, sample_rate, method="phat"):
    n = max(len(signal_1), len(signal_2))
    s1 = np.pad(signal_1, (0, n - len(signal_1)))
    s2 = np.pad(signal_2, (0, n - len(signal_2)))

    fft_s1 = np.fft.rfft(s1)
    fft_s2 = np.fft.rfft(s2)
    cross_spectrum = fft_s1 * np.conjugate(fft_s2)

    if method == "phat":
        cross_spectrum /= (np.abs(cross_spectrum) + 1e-10)
    elif method == "ml":
        mag1 = np.abs(fft_s1)
        mag2 = np.abs(fft_s2)
        cross_spectrum /= (mag1 * mag2 + 1e-10)
    elif method == "scot":
        mag1 = np.abs(fft_s1)
        mag2 = np.abs(fft_s2)
        cross_spectrum /= (np.sqrt(mag1**2 * mag2**2) + 1e-10)
    elif method == "normal":
        pass
    else:
        raise ValueError(f"Unsupported GCC method: {method}")

    corr = np.fft.irfft(cross_spectrum, n)
    max_lag = n // 2
    corr = np.roll(corr, max_lag)
    lags = np.arange(-max_lag, max_lag) / sample_rate
    return lags, corr

def compute_single_tdoa(signal_1, signal_2, sample_rate, method="phat"):
    if len(signal_1) == 0 or len(signal_2) == 0:
        # If signals are empty, return 0
        return 0.0
    lags, corr = gcc(signal_1, signal_2, sample_rate, method)
    idx = np.argmax(corr)
    return lags[idx]
