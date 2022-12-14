import numpy as np
import matplotlib.pyplot as plt
# import scipy.io as sio
import scipy.signal
from scipy import *
import copy
import pymatreader as mr
import pandas as pd
# import sounddevice as sd
from time import sleep
#VIDEO: Welch's method



def za_welch_manual(eegdata, srate):

    # time vector
    N = len(eegdata)
    timevec = np.arange(0, N) / srate


    # window length in seconds*srate
    winlength = int(1 * srate)

    # number of points of overlap
    nOverlap = np.round(srate / 2)

    # window onset times
    winonsets = np.arange(0, int(N - winlength), int(winlength - nOverlap))

    # note: different-length signal needs a different-length Hz vector
    hzW = np.linspace(0, srate / 2, int(np.floor(winlength / 2) + 1))

    # Hann window
    hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, int(winlength))) / 2

    # initialize the power matrix (windows x frequencies)
    eegpowW = np.zeros(len(hzW))

    # loop over frequencies
    for wi in range(0, len(winonsets)):
        # get a chunk of data from this time window
        datachunk = eegdata[winonsets[wi]:winonsets[wi] + winlength]

        # apply Hann taper to data
        datachunk = datachunk * hannw

        # compute its power
        tmppow = np.abs(scipy.fftpack.fft(datachunk) / winlength) ** 2

        # enter into matrix
        eegpowW = eegpowW + tmppow[0:len(hzW)]

    # divide by N
    eegpowW = eegpowW / len(winonsets)

    return eegpowW, hzW

def za_welch_in_build(eegdata, srate):
    ## Python's welch

    # create Hann window
    winsize = int(2 * srate)  # 2-second window
    hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, winsize)) / 2

    # number of FFT points (frequency resolution)
    nfft = srate * 100

    f, welchpow = scipy.signal.welch(eegdata, fs=srate, window=hannw, nperseg=winsize, noverlap=winsize / 4, nfft=nfft)

    return f, welchpow
