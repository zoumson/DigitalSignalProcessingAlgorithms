import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_reflection(data, filt_order, lp_filt_cut_off):
    N = len(data)
    hz = np.linspace(0, 1, N)
    data_fft = np.abs(scipy.fftpack.fft(data)) ** 2

    ## apply a low-pass causal filter

    # generate filter kernel
    fkern = signal.firwin(filt_order, lp_filt_cut_off)

    # zero-phase-shift filter
    fdata = signal.lfilter(fkern, 1, data)  # forward
    fdata = signal.lfilter(fkern, 1, np.flip(fdata, 0))  # reverse
    fdata_no_reflec = np.flip(fdata, 0)  # flip forward

    fdata_no_reflec_fft = np.abs(scipy.fftpack.fft(fdata_no_reflec)) ** 2

    ## now with reflection by filter order

    # reflect the signal
    # data = np.concatenate((np.zeros(100),np.cos(np.linspace(np.pi/2,5*np.pi/2,10)),np.zeros(100)),axis=0)
    reflectdata = np.concatenate((data[filt_order:0:-1], data, data[-1:-1 - filt_order:-1]), axis=0)

    # zero-phase-shift filter on the reflected signal
    reflectdata = signal.lfilter(fkern, 1, reflectdata)
    reflectdata = signal.lfilter(fkern, 1, reflectdata[::-1])
    reflectdata = reflectdata[::-1]

    # now chop off the reflected parts
    fdata_reflec = reflectdata[filt_order:-filt_order]
    fdata_reflec_fft = np.abs(scipy.fftpack.fft(fdata_reflec)) ** 2
    # try again with filtfilt
    fdata_using_filtfilt = signal.filtfilt(fkern, 1, data)

    return data_fft, fkern, fdata_no_reflec, fdata_no_reflec_fft, fdata_reflec, fdata_reflec_fft, fdata_using_filtfilt, hz
