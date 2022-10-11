import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_filt_recursive(data, srate, frex2notch):
    # narrowband filter to remove line noise
    # initialize filtered signal
    datafilt = data
    pnts = len(data)
    # loop over frequencies
    for fi in range(0, len(frex2notch)):
        # create filter kernel using firwin (fir1 in MATLAB)
        frange = [frex2notch[fi] - .5, frex2notch[fi] + .5]
        order = int(150 * (srate / frange[0]))
        order = order + ~order % 2

        # filter kernel
        filtkern = signal.firwin(order, frange, pass_zero=True, fs=srate)

        # visualize the kernel and its spectral response
        # plt.subplot(121)
        # plt.plot(filtkern)
        # plt.title('Time domain')
        #
        # plt.subplot(122)
        # plt.plot(np.linspace(0, srate, 10000), np.abs(scipy.fftpack.fft(filtkern, 10000)) ** 2)
        # plt.xlim([frex2notch[fi] - 30, frex2notch[fi] + 30])
        # plt.title('Frequency domain')
        # plt.show()

        # recursively apply to data
        datafilt = signal.filtfilt(filtkern, 1, datafilt)

    # compute the power spectrum of the filtered signal
    pwrfilt = np.abs(scipy.fftpack.fft(datafilt) / pnts) ** 2

    # compute power spectrum and frequencies vector
    pwr = np.abs(scipy.fftpack.fft(data) / pnts) ** 2
    hz = np.linspace(0, srate, pnts)

    return datafilt, pwrfilt, pwr, hz
