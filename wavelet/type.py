import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy



def za_morlet(npnts, freq, fwhm, timevec):
    # Bandpass filter
    # centered time vector
    # timevec = np.arange(0, npnts) / fs
    # timevec = timevec - np.mean(timevec)

    # for power spectrum
    # hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

    ## Morlet wavelet

    # parameters
    # freq = 4  # peak frequency
    csw = np.cos(2 * np.pi * freq * timevec)  # cosine wave
    # fwhm = .5  # full-width at half-maximum in seconds
    gaussian = np.exp(-(4 * np.log(2) * timevec ** 2) / fwhm ** 2)  # Gaussian

    # Morlet wavelet
    MorletWavelet = csw * gaussian

    # amplitude spectrum
    MorletWaveletPow = np.abs(scipy.fftpack.fft(MorletWavelet) / npnts)

    # return MorletWavelet, MorletWaveletPow, timevec, hz
    return MorletWavelet, MorletWaveletPow

def za_haar(npnts, fs, timevec):
    # Lowpass filter
    ## Haar wavelet
    # centered time vector
    # timevec = np.arange(0, npnts) / fs
    # timevec = timevec - np.mean(timevec)

    # for power spectrum
    # hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))
    # create Haar wavelet
    HaarWavelet = np.zeros(npnts)
    HaarWavelet[np.argmin(timevec ** 2): np.argmin((timevec - .5) ** 2)] = 1
    HaarWavelet[np.argmin((timevec - .5) ** 2): np.argmin((timevec - 1 - 1 / fs) ** 2)] = -1

    # amplitude spectrum
    HaarWaveletPow = np.abs(scipy.fftpack.fft(HaarWavelet) / npnts)
    # return HaarWavelet, HaarWaveletPow, timevec, hz
    return HaarWavelet, HaarWaveletPow

def za_hat(npnts, s, timevec):
    ## Mexican hat wavelet
    # centered time vector
    # timevec = np.arange(0, npnts) / fs
    # timevec = timevec - np.mean(timevec)
    # for power spectrum
    # hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))
    # the wavelet
    # s = .4
    MexicanWavelet = (2 / (np.sqrt(3 * s) * np.pi ** .25)) * (1 - (timevec ** 2) / (s ** 2)) * np.exp(
        (-timevec ** 2) / (2 * s ** 2))

    # amplitude spectrum
    MexicanWaveletPow = np.abs(scipy.fftpack.fft(MexicanWavelet) / npnts)
    # return MexicanWavelet, MexicanWaveletPow, timevec, hz
    return MexicanWavelet, MexicanWaveletPow

def za_dog(npnts, sPos, sNeg, timevec):
    ## Mexican hat wavelet
    # centered time vector
    # timevec = np.arange(0, npnts) / fs
    # timevec = timevec - np.mean(timevec)
    # for power spectrum
    # hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))
    # the wavelet
    ## Difference of Gaussians (DoG)
    # (approximation of Laplacian of Gaussian)

    # define sigmas
    # sPos = .1
    # sNeg = .5

    # create the two GAussians
    gaus1 = np.exp((-timevec ** 2) / (2 * sPos ** 2)) / (sPos * np.sqrt(2 * np.pi))
    gaus2 = np.exp((-timevec ** 2) / (2 * sNeg ** 2)) / (sNeg * np.sqrt(2 * np.pi))

    # their difference is the DoG
    DoG = gaus1 - gaus2

    # amplitude spectrum
    DoGPow = np.abs(scipy.fftpack.fft(DoG) / npnts)


    return DoG, DoGPow
    # return DoG, DoGPow, timevec, hz


def za_morlet_complex(freq, fwhm, timevec):
    # Bandpass filter
    # centered time vector
    # timevec = np.arange(0, npnts) / fs
    # timevec = timevec - np.mean(timevec)

    # for power spectrum
    # hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

    ## Morlet wavelet

    # parameters
    # freq = 4  # peak frequency
    cswSine = np.exp(2*1j*np.pi*freq*timevec)*np.exp(-(4*np.log(2)*timevec**2)/fwhm**2)
    # cswSine = np.cos(2 * np.pi * freq * timevec)  # cosine wave
    # fwhm = .5  # full-width at half-maximum in seconds
    gaussian = np.exp(-(4 * np.log(2) * timevec ** 2) / fwhm ** 2)  # Gaussian

    # Morlet wavelet
    MorletWavelet = cswSine * gaussian

    return MorletWavelet