import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_iir_butter(srate, frange, filt_type, imp_length):
    nyquist = srate / 2
    # create filter coefficients
    fkernB, fkernA = signal.butter(4, np.array(frange) / nyquist, btype=filt_type)
    # power spectrum of filter coefficients
    filtpow = np.abs(scipy.fftpack.fft(fkernB)) ** 2
    hz = np.linspace(0, srate / 2, int(np.floor(len(fkernB) / 2) + 1))

    ## how to evaluate an IIR filter: filter an impulse

    if imp_length % 2 == 0:
        imp_length = imp_length + 1

    # generate the impulse
    impres = np.zeros(imp_length)
    impres[imp_length//2 + 1] = 1

    # apply the filter
    fimp = signal.lfilter(fkernB, fkernA, impres, axis=-1)

    # compute power spectrum
    fimpX = np.abs(scipy.fftpack.fft(fimp)) ** 2
    hz_imp = np.linspace(0, nyquist, int(np.floor(len(impres) / 2) + 1))

    return fkernB, fkernA, hz, filtpow, impres, fimp, fimpX, hz_imp

