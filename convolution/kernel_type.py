import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time


def za_gauss_time_smooth_filt(fwhm, k, srate, sig):
    n = len(sig)
    ## create the Gaussian kernel
    # full-width half-maximum: the key Gaussian parameter
    # fwhm = 25  # in ms

    # normalized time vector in ms
    # k = 100
    gtime = 1000 * np.arange(-k, k) / srate

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)

    # then normalize Gaussian to unit energy
    gauswin = gauswin / np.sum(gauswin)

    fSigTConv = np.zeros(n)
    # implement the running mean filter
    for i in range(k + 1, n - k - 1):
        # each point is the weighted average of k surrounding points
        fSigTConv[i] = np.sum(sig[i - k:i + k] * gauswin)

    ## now repeat in the frequency domain

    # compute N's
    nConv = n + 2 * k + 1 - 1

    # FFTs
    sigX = scipy.fftpack.fft(sig, nConv)
    kernX = scipy.fftpack.fft(gauswin, nConv)
    fSigX = sigX*kernX

    # Power
    sigXP = np.abs(sigX)**2
    kernXP = np.abs(kernX)**2
    fSigXP = np.abs(fSigX)**2


    # IFFT
    filtsigTimeFromFourierAll = np.real(scipy.fftpack.ifft(fSigX))

    # cut wings
    fSigTXNoEdge = filtsigTimeFromFourierAll[k:-k]

    # frequencies vector
    hz = np.linspace(0, srate, nConv)

    return fSigTConv, fSigTXNoEdge, kernX, sigX, fSigX, kernXP, sigXP, fSigXP, hz


def za_gauss_freq_narrow_band_filt(peakf, fwhm, srate, sig):
    ## create Gaussian spectral shape
    # Gaussian parameters (in Hz)
    n = len(sig)

    # peakf = 11
    # fwhm = 5.2

    # vector of frequencies
    hz = np.linspace(0, srate, n)

    # frequency-domain Gaussian
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
    x = hz - peakf  # shifted frequencies
    kernX = np.exp(-.5 * (x / s) ** 2)  # gaussian

    ## now for convolution

    # FFTs
    sigX = scipy.fftpack.fft(sig)
    fSigX = sigX * kernX

    # Power
    sigXP = np.abs(sigX)**2
    fSigXP = np.abs(fSigX)**2

    # IFFT
    fSig = 2 * np.real(scipy.fftpack.ifft(fSigX))
    # fSig = np.real(fSigX)

    # frequencies vector
    hz = np.linspace(0, srate, n)
    return sigX, kernX, fSigX, sigXP, fSigXP, fSig, hz
# VIDEO: Convolution with Planck taper (bandpass filter)
def za_plank_band_pass_filt(srate, sig, fwhm, peakf, eta):
    n = len(sig)
    ## create Planck spectral shape

    # frequencies
    hz = np.linspace(0, srate, n)

    # edge decay, must be between 0 and .5
    # eta = .15

    # spectral parameters
    # fwhm = 13
    # peakf = 20

    # convert fwhm to indices
    mp = np.round(2 * fwhm * n / srate)  # in MATLAB this is np, but np=numpy
    pt = np.arange(1, mp + 1)

    # find center point index
    fidx = np.argmin((hz - peakf) ** 2)

    # define left and right exponentials
    Zl = eta * (mp - 1) * (1 / pt + 1 / (pt - eta * (mp - 1)))
    Zr = eta * (mp - 1) * (1 / (mp - 1 - pt) + 1 / ((1 - eta) * (mp - 1) - pt))

    # create the taper
    offset = mp % 2
    bounds = [np.floor(eta * (mp - 1)) - offset, np.ceil((1 - eta) * (mp - (1 - offset)))]
    plancktaper = np.concatenate((1 / (np.exp(Zl[range(0, int(bounds[0]))]) + 1), np.ones(int(np.diff(bounds) + 1)),
                                  1 / (np.exp(Zr[range(int(bounds[1]), len(Zr) - 1)]) + 1)), axis=0)

    # put the taper inside zeros
    kernX = np.zeros(len(hz))
    pidx = range(int(np.max((0, fidx - np.floor(mp / 2) + 1))), int(fidx + np.floor(mp / 2) - mp % 2 + 1))
    kernX[np.round(pidx)] = plancktaper

    ## now for convolution

    # FFTs
    sigX = scipy.fftpack.fft(sig)
    fSigX = sigX * kernX

    # Power
    sigXP = np.abs(sigX)**2
    fSigXP = np.abs(fSigX)**2

    # IFFT
    fSig = 2 * np.real(scipy.fftpack.ifft(fSigX))

    # frequencies vector
    hz = np.linspace(0, srate, n)


    return fSig, kernX, sigXP, fSigXP, hz

