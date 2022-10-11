import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_filt_high_pass(filtcut, fs, order, imp_length):
    ## now for high-pass filter

    # specify filter cutoff (in Hz)
    # filtcut = 25

    # generate filter coefficients (Butterworth)
    filtb, filta = signal.butter(order, filtcut / (fs / 2), btype='high')


    if imp_length % 2 == 0:
        imp_length = imp_length + 1


    # test impulse response function (IRF)
    impulse = np.zeros(imp_length)
    impulse[imp_length//2 + 1] = 1

    fimpulse = signal.filtfilt(filtb, filta, impulse)
    imptime = np.arange(0, len(impulse)) / fs
    hz = np.linspace(0, fs / 2, 3000)
    imppow = np.abs(scipy.fftpack.fft(fimpulse, 2 * len(hz))) ** 2
    return imptime, impulse, fimpulse, imppow, hz, filta, filtb


def za_filt_high_pass_apply(data, signal1, noise, fs, filtcut, order, imp_length):
    *_, hz, filta, filtb = za_filt_high_pass(filtcut, fs, order, imp_length)
    N = len(data)
    # now filter the data and compare against the original
    fdata = signal.filtfilt(filtb, filta, data)

    ### power spectra of original and filtered signal
    signalX = np.abs(scipy.fftpack.fft(signal1) / N) ** 2
    fdataX = np.abs(scipy.fftpack.fft(fdata) / N) ** 2
    signal1_fft = np.abs(scipy.fftpack.fft(signal1) / N) ** 2
    noise_fft = np.abs(scipy.fftpack.fft(noise) / N) ** 2
    return fdata, signalX, fdataX, signal1_fft, noise_fft
