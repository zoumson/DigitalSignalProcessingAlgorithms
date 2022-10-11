import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_filt_narrow_band(samprate, filtorder, lower_bnd, upper_bnd, lower_trans, upper_trans):
    # define filter parameters
    # lower_bnd = 10  # Hz
    # upper_bnd = 18  # Hz
    #
    # lower_trans = .1
    # upper_trans = .4

    filter_shape = [0, 0, 1, 1, 0, 0]
    filter_freqs = [0, lower_bnd * (1 - lower_trans), lower_bnd, upper_bnd, \
                    upper_bnd + upper_bnd * upper_trans, samprate / 2]

    filterkern = signal.firls(filtorder, filter_freqs, filter_shape, fs=samprate)
    hz = np.linspace(0, samprate / 2, int(np.floor(len(filterkern) / 2) + 1))
    filterpow = np.abs(scipy.fftpack.fft(filterkern)) ** 2

    return filterkern, filterpow, filter_freqs, filter_shape, hz


def za_filt_narrow_band_apply(data, samprate, filtorder, lower_bnd, upper_bnd, lower_trans, upper_trans):
    filterkern, filterpow, filter_freqs, filter_shape, hz = za_filt_narrow_band(samprate, filtorder, lower_bnd, \
                                                                                upper_bnd, lower_trans, upper_trans)
    ## now apply to random noise data

    filtnoise = signal.filtfilt(filterkern, 1, np.random.randn(samprate * 4))

    timevec = np.arange(0, len(filtnoise)) / samprate
    timevec = np.arange(0, len(filtnoise)) / samprate

    noisepower = np.abs(scipy.fftpack.fft(filtnoise)) ** 2

    return timevec, filtnoise
