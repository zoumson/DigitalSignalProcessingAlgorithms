import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy



def za_reflect_entire_sig(signal1, filtN, filt_cut_off):
    dataN = len(signal1)
    # create filter kernel
    fkern = signal.firwin(filtN, filt_cut_off, pass_zero=True)
    data_fft = np.abs(scipy.fftpack.fft(signal1)) ** 2
    # apply filter kernel to data
    fdata_direct = signal.filtfilt(fkern, 1, signal1)
    fdata_direct_fft = np.abs(scipy.fftpack.fft(fdata_direct)) ** 2
    # reflect the signal
    signalRefl = np.concatenate((signal1[::-1], signal1, signal1[::-1]), axis=0)
    # apply filter kernel to data
    fdataR = signal.filtfilt(fkern, 1, signalRefl)

    # and cut off edges
    fdata_entire_reflect = fdataR[dataN:-dataN]
    fdata_entire_reflect_fft = np.abs(scipy.fftpack.fft(fdata_entire_reflect)) ** 2
    hz = np.linspace(0, 1, dataN)
    return fkern, data_fft, fdata_direct, fdata_direct_fft, fdata_entire_reflect, fdata_entire_reflect_fft, hz






