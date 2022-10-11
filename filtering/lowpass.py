import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy



def za_low_pass_filt(y , timevec, fcutoff, transw, order, fs):
    npnts = len(timevec)
    # power spectrum of signal
    yX = np.abs(scipy.fftpack.fft(y) / npnts) ** 2
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))


    shape = [1, 1, 0, 0]
    frex = [0, fcutoff, fcutoff + fcutoff * transw, fs / 2]
    # filter kernel
    filtkern = signal.firls(order, frex, shape, fs=fs)
    # its power spectrum
    filtkernX = np.abs(scipy.fftpack.fft(filtkern, npnts)) ** 2

    ### now apply the filter to the data
    yFilt = signal.filtfilt(filtkern, 1, y)

    ### power spectra of original and filtered signal
    yOrigX = np.abs(scipy.fftpack.fft(y) / npnts) ** 2
    yFiltX = np.abs(scipy.fftpack.fft(yFilt) / npnts) ** 2

    return yX,hz, filtkern, frex, shape, filtkernX, yFilt, yOrigX, yFiltX
