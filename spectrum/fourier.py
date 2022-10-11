import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.fftpack
import scipy.signal
import scipy.io.wavfile
import copy


def za_transform_freq_amp(sig, srate):
    npnts = srate * 2  # 2 seconds
    time = np.arange(0, npnts) / srate
    # amplitude spectrum via Fourier transform
    signalX = scipy.fftpack.fft(sig)
    signalAmp = 2 * np.abs(signalX) / npnts
    # vector of frequencies in Hz
    hz = np.linspace(0, srate / 2, int(np.floor(npnts / 2) + 1))

    return signalX, signalAmp, time, hz


def za_transform_pow(sig):
    N = len(sig)
    # possible normalizations...
    sig = sig - np.mean(sig)
    # power
    sigPow = np.abs(scipy.fftpack.fft(sig) / N) ** 2

    return sigPow



