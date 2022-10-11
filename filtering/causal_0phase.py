import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy



def za_causal_low_pass(data, num_taps, lp_cut_off):
    n = len(data)
    data_fft = np.abs(scipy.fftpack.fft(data))
    ## apply a low-pass causal filter

    # note: frequency listed as fraction of Nyquist (not sampling rate!)
    fkern = signal.firwin(num_taps, lp_cut_off)
    fdata = signal.lfilter(fkern, 1, data)

    return data_fft, fkern, fdata

def za_0_phase_low_pass(data, num_taps, lp_cut_off):
    n = len(data)
    data_fft = np.abs(scipy.fftpack.fft(data))
    ## apply a low-pass causal filter

    # note: frequency listed as fraction of Nyquist (not sampling rate!)
    fkern = signal.firwin(num_taps, lp_cut_off)
    fdata = signal.lfilter(fkern, 1, data)

    # flip the signal backwards
    fdataFlip = fdata[::-1]

    fdataFlip_fft = np.abs(scipy.fftpack.fft(fdataFlip))

    # filter the flipped signal
    ffdataFlip = signal.lfilter(fkern, 1, fdataFlip)

    # finally, flip the double-filtered signal
    ffdataFlipFlip = ffdataFlip[::-1]

    return data_fft, fkern, fdata, fdataFlip, fdataFlip_fft, ffdataFlip, ffdataFlipFlip


