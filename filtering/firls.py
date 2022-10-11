import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_filt_firls_normal(order, frex, shape, srate):
    # order must be odd
    if order % 2 == 0:
        order += 1
    # filter kernel
    filtkern = signal.firls(order, frex, shape, fs=srate)

    # compute the power spectrum of the filter kernel
    filtpow = np.abs(scipy.fftpack.fft(filtkern)) ** 2
    # compute the frequencies vector and remove negative frequencies
    hz = np.linspace(0, srate / 2, int(np.floor(len(filtkern) / 2) + 1))
    filtpow = filtpow[0:len(hz)]

    return filtkern, filtpow, hz