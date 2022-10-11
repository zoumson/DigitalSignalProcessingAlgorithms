import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_filt_sinc_window(srate, time, f):
    pnts = len(time)
    sincfilt = np.sin(2 * np.pi * f * time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt / np.sum(sincfilt)

    # windowed sinc filter
    sincfiltW = sincfilt * np.hanning(pnts)

    hz = np.linspace(0, srate / 2, int(np.floor(pnts / 2) + 1))
    pw_non_window = np.abs(scipy.fftpack.fft(sincfilt))
    pw_with_window = np.abs(scipy.fftpack.fft(sincfiltW))

    return sincfilt, sincfiltW, pw_non_window, pw_with_window, hz


def za_filt_sinc_window_apply(srate, time, f, data):
    sincfilt, sincfiltW, pw_non_window, pw_with_window, hz = za_filt_sinc_window(srate, time, f)
    pnts = len(time)
    # reflection
    datacat = np.concatenate((data, data[::-1]), axis=0)

    # apply filter (zero-phase-shift)
    dataf = signal.lfilter(sincfiltW, 1, datacat)
    dataf = signal.lfilter(sincfiltW, 1, dataf[::-1])

    # flip forwards and remove reflected points
    dataf = dataf[-1:pnts - 1:-1]

    # compute spectra of original and filtered signals
    powOrig_fft = np.abs(scipy.fftpack.fft(data) / pnts) ** 2
    powFilt_fft = np.abs(scipy.fftpack.fft(dataf) / pnts) ** 2
    hz = np.linspace(0, srate / 2, int(np.floor(pnts / 2) + 1))
    return dataf, powOrig_fft, powFilt_fft, hz

def za_filt_sinc_hann_wind(sincfilt, pnts):
    # with Hann taper
    # sincfiltW[0,:] = sincfilt * np.hanning(pnts)
    hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, pnts)) / 2
    hannw_sincfilt = sincfilt * hannw
    hannw_pw = np.abs(scipy.fftpack.fft(hannw_sincfilt))
    return hannw_sincfilt, hannw_pw

def za_filt_sinc_hamming_wind(sincfilt, pnts):
    # sincfiltW[1,:] = sincfilt * np.hamming(pnts)
    hammingw = .54 - .46 * np.cos(2 * np.pi * np.linspace(0, 1, pnts))
    hamming_sincfilt = sincfilt * hammingw
    hamming_pw = np.abs(scipy.fftpack.fft(hamming_sincfilt))
    return hamming_sincfilt, hamming_pw

def za_filt_sinc_gauss_wind(sincfilt, time):
    gauss_sincfilt = sincfilt * np.exp(-time**2)
    gauss_pw = np.abs(scipy.fftpack.fft(gauss_sincfilt))
    return gauss_sincfilt, gauss_pw

