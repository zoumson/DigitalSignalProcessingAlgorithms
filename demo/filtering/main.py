import filtering.firls as zffz
import filtering.firwin as zffw
import filtering.firwin as zffw
import filtering.butter as zfb
import filtering.causal_0phase as zfc0p
import filtering.reflection as zfr
import filtering.signal_kernel_length as zfrs
import filtering.lowpass as zlp
import filtering.window_sinc as zfws
import filtering.highpass as zfh
import filtering.narrow_band as znb
import filtering.linenoise as znl
import filtering.rolloff as zf3dbr

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy
import pymatreader as mr
import pandas as pd


def za_filt_firls_1():
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    transw = .1
    order = int(5 * srate / frange[0])

    # order must be odd
    if order % 2 == 0:
        order += 1

    # define filter shape
    shape = [0, 0, 1, 1, 0, 0]
    frex = [0, frange[0] - frange[0] * transw, frange[0], frange[1], frange[1] + frange[1] * transw, nyquist]

    filtkern, filtpow, hz = zffz.za_filt_firls_normal(order, frex, shape, srate)

    # time-domain filter kernel
    plt.plot(filtkern)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')
    plt.show()

    # plot amplitude spectrum of the filter kernel
    plt.plot(hz, filtpow, 'ks-', label='Actual')
    plt.plot(frex, shape, 'ro-', label='Ideal')
    plt.xlim([0, frange[0] * 4])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firls)')
    plt.show()


def za_filt_firls_2():
    ## effects of the filter kernel order
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    transw = .1

    # define filter shape
    shape = [0, 0, 1, 1, 0, 0]
    frex = [0, frange[0] - frange[0] * transw, frange[0], frange[1], frange[1] + frange[1] * transw, nyquist]

    ## effects of the filter kernel order

    # range of orders
    ordersF = (1 * srate / frange[0]) / (srate / 1000)
    ordersL = (15 * srate / frange[0]) / (srate / 1000)

    orders = np.round(np.linspace(ordersF, ordersL, 10))

    # initialize
    fkernX = np.zeros((len(orders), 1000))
    hz = np.linspace(0, srate, 1000)

    for oi in range(0, len(orders)):
        # make sure order is odd-length
        ord2use = orders[oi] + (1 - orders[oi] % 2)

        # create filter kernel
        fkern = signal.firls(ord2use, frex, shape, fs=srate)

        # take its FFT
        fkernX[oi, :] = np.abs(scipy.fftpack.fft(fkern, 1000)) ** 2

        # show in plot
        time = np.arange(0, ord2use) / srate
        time = time - np.mean(time)
        plt.plot(time, fkern + .01 * oi)

    plt.xlabel('Time (ms)')
    plt.title('Filter kernels with different orders')
    plt.show()

    plt.plot(hz, fkernX.T)
    plt.plot(frex, shape, 'k')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (firls)')
    plt.show()

    plt.plot(hz, 10 * np.log10(fkernX.T))
    plt.xlim([0, 100])
    plt.title('Same as above but logscale')
    plt.show()


def za_filt_firls_3():
    ## effects of the filter transition width
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]

    # define filter shape
    shape = [0, 0, 1, 1, 0, 0]

    # range of transitions
    transwidths = np.linspace(.01, .4, 10)

    # initialize
    fkernX = np.zeros((len(transwidths), 1000))
    hz = np.linspace(0, srate, 1000)

    for ti in range(0, len(transwidths)):
        # create filter kernel
        frex = [0, frange[0] - frange[0] * transwidths[ti], frange[0], frange[1],
                frange[1] + frange[1] * transwidths[ti], nyquist]
        fkern = signal.firls(401, frex, shape, fs=srate)
        n = len(fkern)

        # take its FFT
        fkernX[ti, :] = np.abs(scipy.fftpack.fft(fkern, 1000)) ** 2

        # show in plot
        time = np.arange(0, 401) / srate
        time = time - np.mean(time)
        plt.plot(time, fkern + .01 * ti)

    plt.xlabel('Time (ms)')
    plt.title('Filter kernels with different transition widths')
    plt.show()

    plt.plot(hz, fkernX.T)
    plt.plot(frex, shape, 'k')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (firls)')
    plt.show()

    plt.plot(hz, 10 * np.log10(fkernX.T))
    plt.plot(frex, shape, 'k')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Same as above but log')
    plt.show()


def za_filt_firwin_1():
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    transw = .1
    order = int(5 * srate / frange[0])

    # force odd order
    if order % 2 == 0:
        order += 1

    ### --- NOTE: Python's firwin corresponds to MATLAB's fir1 --- ###

    # filter kernel
    filtkern, filtpow, hz = zffw.za_filt_firwin_normal(order, frange, srate)

    # time-domain filter kernel
    plt.plot(filtkern)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firwin)')
    plt.show()
    #
    # # compute the power spectrum of the filter kernel
    # filtpow = np.abs(scipy.fftpack.fft(filtkern)) ** 2
    # # compute the frequencies vector and remove negative frequencies
    # hz = np.linspace(0, srate / 2, int(np.floor(len(filtkern) / 2) + 1))
    # filtpow = filtpow[0:len(hz)]

    # plot amplitude spectrum of the filter kernel
    plt.plot(hz, filtpow, 'ks-', label='Actual')
    plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'ro-', label='Ideal')
    plt.xlim([0, frange[0] * 4])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firwin)')
    plt.show()

    # Same as above but logarithmically scaled
    plt.plot(hz, 10 * np.log10(filtpow), 'ks-', label='Actual')
    plt.plot([frange[0], frange[0]], [-100, 5], 'ro-', label='Ideal')
    plt.xlim([0, frange[0] * 4])
    plt.ylim([-80, 5])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firwin)')
    plt.show()


def za_filt_firwin_2():
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    transw = .1
    ## effects of the filter kernel order

    # range of orders
    orders = np.round(np.linspace((srate / frange[0]) / (srate / 1000), (15 * srate / frange[0]) / (srate / 1000), 10))
    # initialize
    fkernX = np.zeros((len(orders), 1000))
    hz = np.linspace(0, srate, 1000)

    for oi in range(0, len(orders)):
        # make sure order is odd-length
        ord2use = orders[oi] + (1 - orders[oi] % 2)

        # create filter kernel
        fkern = signal.firwin(int(ord2use), frange, fs=srate, pass_zero=False)

        # take its FFT
        fkernX[oi, :] = np.abs(scipy.fftpack.fft(fkern, 1000)) ** 2

        # show in plot
        time = np.arange(0, ord2use) / srate
        time = time - np.mean(time)
        plt.plot(time, fkern + .01 * oi)

    plt.xlabel('Time (ms)')
    plt.title('Filter kernels with different orders')
    plt.show()

    plt.plot(hz, fkernX.T)
    plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'k')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (firwin)')
    plt.show()

    plt.plot(hz, 10 * np.log10(fkernX.T))
    plt.xlim([0, 100])
    plt.title('Same as above but logscale')
    plt.show()


def za_filt_butter_1():
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    filt_type = 'bandpass'
    imp_length = 1001

    fkernB, fkernA, hz, filtpow, impres, fimp, fimpX, hz_imp = zfb.za_iir_butter(srate, frange, filt_type, imp_length)

    # plotting
    plt.subplot(121)
    plt.plot(fkernB * 1e5, 'ks-', label='B')
    plt.plot(fkernA, 'rs-', label='A')
    plt.xlabel('Time points')
    plt.ylabel('Filter coeffs.')
    plt.title('Time-domain filter coefs')
    plt.legend()

    plt.subplot(122)
    plt.stem(hz, filtpow[0:len(hz)], 'ks-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power spectrum filter coeffs.')
    plt.show()

    # plot impulse response
    plt.plot(impres, 'k', label='Impulse')
    plt.plot(fimp, 'r', label='Filtered')
    plt.xlim([1, len(impres)])
    plt.ylim([-.06, .06])
    plt.legend()
    plt.xlabel('Time points (a.u.)')
    plt.title('Filtering an impulse')
    plt.show()

    plt.plot(hz_imp, fimpX[0:len(hz_imp)], 'ks-')
    plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'r')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()

    plt.plot(hz_imp, 10 * np.log10(fimpX[0:len(hz_imp)]), 'ks-')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()


def za_filt_butter_2():
    # filter parameters
    srate = 1024  # hz
    nyquist = srate / 2
    frange = [20, 45]
    ## effects of order parameter

    filt_type = 'bandpass'
    imp_length = 1001

    fkernB, fkernA, hz, filtpow, impres, fimp, fimpX, hz_imp = zfb.za_iir_butter(srate, frange, filt_type, imp_length)

    orders = range(2, 8)

    fkernX = np.zeros((len(orders), imp_length))
    hz = np.linspace(0, srate, imp_length)

    # loop over orders
    for oi in range(0, len(orders)):
        # create filter kernel
        fkernB, fkernA = signal.butter(orders[oi], np.array(frange) / nyquist, btype='bandpass')
        n = len(fkernB)

        # filter the impulse response and take its power
        fimp = signal.lfilter(fkernB, fkernA, impres, axis=-1)
        fkernX[oi, :] = np.abs(scipy.fftpack.fft(fimp)) ** 2

        # show in plot
        time = np.arange(0, len(fkernB)) / srate
        time = time - np.mean(time)
        plt.subplot(121)
        plt.plot(time, scipy.stats.zscore(fkernB) + oi)
        plt.title('Filter coefficients (B)')

        plt.subplot(122)
        plt.plot(time, scipy.stats.zscore(fkernA) + oi)
        plt.title('Filter coefficients (A)')

    plt.show()

    # plot the spectra
    plt.plot(hz, fkernX.T)
    plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'r')
    plt.xlim([0, 100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()

    # in log space
    plt.plot(hz, 10 * np.log10(fkernX.T))
    plt.xlim([0, 100])
    plt.ylim([-80, 2])
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()


def za_filt_causal():
    # create a simple signal
    data = np.concatenate((np.zeros(100), np.cos(np.linspace(np.pi / 2, 5 * np.pi / 2, 10)), np.zeros(100)), axis=0)
    n = len(data)
    num_taps = 51
    lp_cut_off = .6
    data_fft, fkern, fdata = zfc0p.za_causal_low_pass(data, num_taps, lp_cut_off)
    # plot it and its power spectrum
    plt.subplot(121)
    plt.plot(range(0, n), data, 'ko-')
    plt.xlim([0, n + 1])
    plt.title('Original signal')
    plt.xlabel('Time points (a.u.)')

    plt.subplot(122)
    plt.plot(np.linspace(0, 1, n), data_fft, 'ko-')
    plt.xlim([0, .5])
    plt.xlabel('Frequency (norm.)')
    plt.ylabel('Energy')
    plt.title('Frequency-domain signal representation')
    plt.show()
    ## apply a low-pass causal filter

    plt.plot(range(0, n), data, label='Original')
    plt.plot(range(0, n), fdata, label='Forward filtered')
    plt.legend()
    plt.show()


def za_0_phase():
    # create a simple signal
    data = np.concatenate((np.zeros(100), np.cos(np.linspace(np.pi / 2, 5 * np.pi / 2, 10)), np.zeros(100)), axis=0)
    n = len(data)
    num_taps = 51
    lp_cut_off = .6

    # data_fft, fkern, fdata = zfc0p.za_causal_low_pass(data, num_taps, lp_cut_off)
    data_fft, fkern, fdata, fdataFlip, fdataFlip_fft, ffdataFlip, ffdataFlipFlip = zfc0p.za_0_phase_low_pass(data,
                                                                                                             num_taps,
                                                                                                             lp_cut_off)
    # plot it and its power spectrum
    plt.subplot(121)
    plt.plot(range(0, n), data, 'ko-')
    plt.xlim([0, n + 1])
    plt.title('Original signal')
    plt.xlabel('Time points (a.u.)')

    plt.subplot(122)
    plt.plot(np.linspace(0, 1, n), data_fft, 'ko-')
    plt.xlim([0, .5])
    plt.xlabel('Frequency (norm.)')
    plt.ylabel('Energy')
    plt.title('Frequency-domain signal representation')
    plt.show()
    ## apply a low-pass causal filter

    plt.plot(range(0, n), data, label='Original')
    plt.plot(range(0, n), fdata, label='Forward filtered')
    plt.title('Time Domain')
    plt.legend()
    plt.show()

    # and show its spectrum
    plt.plot(np.linspace(0, 1, n), data_fft, 'ko-', label='Original')
    plt.plot(np.linspace(0, 1, n), fdataFlip_fft, 'r', label='Forward filtered')
    plt.title('Frequency Domain')
    plt.xlim([0, .5])
    plt.legend()
    plt.show()

    # filter the flipped signal
    plt.plot(range(0, n), data, label='Original')
    plt.plot(range(0, n), ffdataFlip, label='Backward filtered')
    plt.title('Time Domain')
    plt.legend()
    plt.show()

    # finally, flip the double-filtered signal
    plt.plot(range(0, n), data, label='Original')
    plt.plot(range(0, n), ffdataFlipFlip, label='Zero-phase filtered')
    plt.title('Time Domain')
    plt.legend()
    plt.show()


def za_filt_filtfilt():
    N = 500
    hz = np.linspace(0, 1, N)
    gx = np.exp(-(4 * np.log(2) * (hz - .1) / .1) ** 2) * N / 2
    data = np.real(scipy.fftpack.ifft(gx * np.exp(1j * np.random.rand(N) * 2 * np.pi)))
    data = data + np.random.randn(N)
    filt_order = 151
    lp_filt_cut_off = .6
    data_fft, fkern, fdata_no_reflec, fdata_no_reflec_fft, fdata_reflec, fdata_reflec_fft, fdata_using_filtfilt, hz = zfr.za_reflection(
        data,
        filt_order, lp_filt_cut_off)
    # plot it and its power spectrum
    plt.plot(range(0, N), data, 'k')
    plt.title('Original signal')
    plt.xlabel('Time (a.u.)')
    plt.show()

    plt.plot(hz, data_fft, 'k')
    plt.xlim([0, .5])
    plt.xlabel('Frequency (norm.)')
    plt.ylabel('Energy')
    plt.title('Frequency-domain signal representation')
    plt.show()

    # plot the original signal and filtered version
    plt.subplot(121)
    plt.plot(range(0, N), data, 'k', label='Original')
    plt.plot(range(0, N), fdata_no_reflec, 'm', label='Filtered, no reflection')
    plt.title('Time domain')
    plt.legend()

    # power spectra
    plt.subplot(122)
    plt.plot(hz, data_fft, 'k', label='Original')
    plt.plot(hz, fdata_no_reflec_fft, 'm', label='Filtered, no reflection')
    plt.title('Frequency domain')
    plt.xlim([0, .5])
    plt.legend()
    plt.show()

    # and plot
    plt.plot(range(0, N), data, 'k', label='original')
    plt.plot(range(0, N), fdata_reflec, 'm', label='filtered reflected')
    plt.plot(range(0, N), fdata_using_filtfilt, 'b', label='filtered filtfilt')
    plt.xlabel('Time (a.u.)')
    plt.title('Time domain')
    plt.legend()
    plt.show()

    # spectra
    plt.plot(hz, data_fft, 'k', label='Original')
    plt.plot(hz, fdata_reflec_fft, 'm', label='Filtered')
    plt.legend()
    plt.xlim([0, .5])
    plt.xlabel('Frequency (norm.)')
    plt.ylabel('Energy')
    plt.title('Frequency domain')
    plt.show()


def za_kern_sig_len():
    # parameters
    dataN = 100000
    filtN = 501
    # generate data
    signal1 = np.random.randn(dataN)
    filt_cut_off = .01
    fkern, data_fft, fdata_direct, fdata_direct_fft, fdata_entire_reflect, fdata_entire_reflect_fft, hz = zfrs.za_reflect_entire_sig(
        signal1, filtN, filt_cut_off)
    N = dataN
    # and plot
    plt.plot(range(0, N), signal1, 'k', label='original')
    plt.plot(range(0, N), fdata_direct, 'm', label='filtered direct')
    plt.plot(range(0, N), fdata_entire_reflect, 'b', label='filtered with full reflection')
    plt.xlabel('Time (a.u.)')
    plt.title('Time domain')
    plt.legend()
    plt.show()

    # spectra
    plt.plot(hz, data_fft, 'k', label='Original')
    plt.plot(hz, fdata_direct_fft, 'm', label='Filtered direct')
    plt.plot(hz, fdata_entire_reflect_fft, 'm', label='Filtered with full reflection')
    plt.legend()
    plt.xlim([0, .5])
    plt.xlabel('Frequency (norm.)')
    plt.ylabel('Energy')
    plt.title('Frequency domain')
    plt.show()


def za_low_pass():
    # simulation parameters
    fs = 350  # hz
    timevec = np.arange(0, fs * 7 - 1) / fs

    npnts = len(timevec)

    # generate signal
    yOrig = np.cumsum(np.random.randn(npnts))
    y = yOrig + 50 * np.random.randn(npnts) + 40 * np.sin(2 * np.pi * 50 * timevec)

    ## now for lowpass filter

    fcutoff = 30
    transw = .2
    order = np.round(7 * fs / fcutoff) + 1

    yX, hz, filtkern, frex, shape, filtkernX, yFilt, yOrigX, yFiltX = zlp.za_low_pass_filt(y, timevec, fcutoff, transw,
                                                                                           order, fs)

    # plot the data
    plt.plot(timevec, y, label='Measured')
    plt.plot(timevec, yOrig, label='Original')
    plt.xlabel('Time (sec.)')
    plt.ylabel('Power')
    plt.title('Time domain')
    plt.legend()
    plt.show()

    # plot its power spectrum
    plt.plot(hz, yX[0:len(hz)], 'k')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Frequency domain')
    plt.yscale('log')
    plt.show()

    plt.plot(np.arange(-order / 2, order / 2) / fs, filtkern, 'k')
    plt.xlabel('Time (s)')
    plt.title('Filter kernel')
    plt.show()

    plt.plot(np.array(frex), shape, 'r')
    plt.plot(hz, filtkernX[:len(hz)], 'k')
    plt.xlim([0, 60])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Filter kernel spectrum')
    plt.show()

    plt.plot(timevec, y, label='Signal')
    plt.plot(timevec, yFilt, label='Filtered')
    plt.legend()
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.show()

    plt.plot(hz, yOrigX[:len(hz)], label='Signal')
    plt.plot(hz, yFiltX[:len(hz)], label='Filtered')
    plt.xlim([0, fs / 5])
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.show()


def za_sinc_wind():
    # simulation params
    srate = 1000
    time = np.arange(-4, 4, 1 / srate)
    pnts = len(time)
    # create sinc function
    f = 8
    sincfilt, sincfiltW, pw_non_window, pw_with_window, hz = zfws.za_filt_sinc_window(srate, time, f)

    # plot the sinc filter
    plt.subplot(121)
    plt.plot(time, sincfilt, 'k')
    plt.xlabel('Time (s)')
    plt.title('Non-windowed sinc function')

    # plot the power spectrum
    plt.subplot(122)
    hz = np.linspace(0, srate / 2, int(np.floor(pnts / 2) + 1))
    pw = np.abs(scipy.fftpack.fft(sincfilt))
    plt.plot(hz, pw[:len(hz)], 'k')
    plt.xlim([0, f * 3])
    plt.yscale('log')
    plt.plot([f, f], [0, 1], 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.show()

    # now plot the windowed sinc filter
    plt.subplot(121)
    plt.plot(time, sincfiltW, 'k')
    plt.xlabel('Time (s)')
    plt.title('Windowed sinc function')

    plt.subplot(122)
    pw = np.abs(scipy.fftpack.fft(sincfiltW))
    plt.plot(hz, pw[:len(hz)])
    plt.xlim([0, f * 3])
    plt.yscale('log')
    plt.plot([f, f], [0, 1], 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.show()


def za_sinc_wind_app():
    # simulation params
    f = 8
    srate = 1000
    time = np.arange(-4, 4, 1 / srate)
    pnts = len(time)
    ## apply the filter to noise

    # generate data as integrated noise
    data = np.cumsum(np.random.randn(pnts))

    dataf, powOrig_fft, powFilt_fft, hz = zfws.za_filt_sinc_window_apply(srate, time, f, data)

    # plot
    plt.plot(time, data, label='Original')
    plt.plot(time, dataf, label='Windowed-sinc filtred')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # plot original and filtered spectra
    plt.plot(hz, powOrig_fft[:len(hz)], label='Original')
    plt.plot(hz, powFilt_fft[:len(hz)], label='Windowed-sinc filtred')
    plt.xlim([0, 40])
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.show()


def za_sinc_all_3_wind():
    # simulation params
    srate = 1000
    time = np.arange(-4, 4, 1 / srate)
    pnts = len(time)
    f = 8
    ## with different windowing functions

    sincfiltW = np.zeros((3, pnts))
    pw = np.zeros((3, pnts))

    tapernames = ['Hann', 'Hamming', 'Gauss']

    # sincfilt, *_ = zfws.za_filt_sinc_window(srate, time, f)
    res = zfws.za_filt_sinc_window(srate, time, f)
    sincfilt, hz = res[0], res[-1]
    sincfilt = zfws.za_filt_sinc_window(srate, time, f)[0]
    sincfiltW[0, :], pw[0, :] = zfws.za_filt_sinc_hann_wind(sincfilt, pnts)
    sincfiltW[1, :], pw[1, :] = zfws.za_filt_sinc_hamming_wind(sincfilt, pnts)
    sincfiltW[2, :], pw[2, :] = zfws.za_filt_sinc_gauss_wind(sincfilt, time)

    # plot them

    for filti in range(0, len(sincfiltW)):
        plt.subplot(121)
        plt.plot(time, sincfiltW[filti, :], label=tapernames[filti])
        # print(pw[filti, :])
        plt.subplot(122)

        plt.plot(hz, pw[filti, :len(hz)], label=tapernames[filti])
        plt.xlim([f - 3, f + 10])
        plt.yscale('log')

    plt.legend()
    plt.show()


def za_high_pass_filt_generate():
    fs = 1000
    ## now for high-pass filter

    # specify filter cutoff (in Hz)
    filtcut = 25
    order = 7
    imp_length = 1001
    imptime, impulse, fimpulse, imppow, hz, *_ = zfh.za_filt_high_pass(filtcut, fs, order, imp_length)

    # plot impulse and IRF
    plt.subplot(121)
    plt.plot(imptime, impulse, label='Impulse')
    plt.plot(imptime, fimpulse / np.max(fimpulse), label='Impulse response')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title('Time domain filter characteristics')

    # plot spectrum of IRF
    plt.subplot(122)
    hz = np.linspace(0, fs / 2, 3000)
    imppow = np.abs(scipy.fftpack.fft(fimpulse, 2 * len(hz))) ** 2
    plt.plot(hz, imppow[:len(hz)], 'k')

    # plt.plot([filtcut, filtcut], [0, 1], 'r--')
    plt.axvline(x=25, linestyle='--', color='r')
    plt.xlim([0, 60])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency domain filter characteristics')
    plt.show()


def za_high_pass_filt_apply():
    # generate 1/f noise
    N = 8000
    fs = 1000
    as1 = np.random.rand(N) * np.exp(-np.arange(0, N) / 200)
    fc = as1 * np.exp(1j * 2 * np.pi * np.random.rand(len(as1)))
    noise = np.real(scipy.fftpack.ifft(fc)) * N

    ### create frequency-domain Gaussian
    hz = np.linspace(0, fs, N)
    s = 4 * (2 * np.pi - 1) / (4 * np.pi);  # normalized width
    x = hz - 30  # shifted frequencies
    fg = np.exp(-.5 * (x / s) ** 2)  # gaussian

    fc = np.random.rand(N) * np.exp(1j * 2 * np.pi * np.random.rand(N))
    fc = fc * fg

    # generate signal from Fourier coefficients, and add noise
    signal1 = np.real(scipy.fftpack.ifft(fc)) * N
    data = signal1 + noise
    time = np.arange(0, N) / fs
    fs = 1000
    ## now for high-pass filter

    # specify filter cutoff (in Hz)
    filtcut = 25
    order = 7
    imp_length = 1001
    fdata, signalX, fdataX, signal1_fft, noise_fft = zfh.za_filt_high_pass_apply(data, signal1, noise, fs, filtcut,
                                                                                 order, imp_length)
    ### plot the data
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Data = signal + noise')
    plt.show()

    plt.plot(hz, signal1_fft, label='signal')
    plt.plot(hz, noise_fft, label='noise')
    plt.legend()
    plt.xlim([0, 100])
    plt.title('Frequency domain representation of signal and noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()

    # now filter the data and compare against the original
    plt.plot(time, signal1, label='Original')
    plt.plot(time, fdata, label='Filtered')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time domain')
    plt.show()

    ### power spectra of original and filtered signal
    # signalX = np.abs(scipy.fftpack.fft(signal1) / N) ** 2
    # fdataX = np.abs(scipy.fftpack.fft(fdata) / N) ** 2
    # hz = np.linspace(0, fs, N)

    plt.plot(hz, signalX[:len(hz)], label='original')
    plt.plot(hz, fdataX[:len(hz)], label='filtered')
    plt.xlim([20, 60])
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Frequency domain')
    plt.show()


def za_narrow_band_1():
    # define filter parameters
    lower_bnd = 10  # Hz
    upper_bnd = 18  # Hz

    lower_trans = .1
    upper_trans = .4

    samprate = 2048  # Hz
    filtorder = 4 * np.round(samprate / lower_bnd) + 1

    filterkern, filterpow, filter_freqs, filter_shape, hz = znb.za_filt_narrow_band(samprate, filtorder, lower_bnd,
                                                                                    upper_bnd, lower_trans, upper_trans)

    # let's see it
    plt.subplot(121)
    plt.plot(filterkern)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')

    # plot amplitude spectrum of the filter kernel
    plt.subplot(122)
    plt.plot(hz, filterpow[:len(hz)], 'ks-', label='Actual')
    plt.plot(filter_freqs, filter_shape, 'ro-', label='Ideal')

    # make the plot look nicer
    plt.xlim([0, upper_bnd + 20])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response')
    plt.legend()
    plt.show()


def za_narrow_band_2():
    # define filter parameters
    lower_bnd = 10  # Hz
    upper_bnd = 18  # Hz

    lower_trans = .1
    upper_trans = .4

    samprate = 2048  # Hz
    filtorder = 4 * np.round(samprate / lower_bnd) + 1
    data = np.random.randn(samprate * 4)
    timevec, filtnoise = znb.za_filt_narrow_band_apply(data, samprate, filtorder, lower_bnd, upper_bnd, lower_trans,
                                                       upper_trans)

    # plot time series
    plt.subplot(121)
    plt.plot(timevec, filtnoise)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Amplitude')
    plt.title('Filtered noise')

    # plot power spectrum
    noisepower = np.abs(scipy.fftpack.fft(filtnoise)) ** 2
    plt.subplot(122)
    plt.plot(np.linspace(0, samprate, len(noisepower)), noisepower, 'k')
    plt.xlim([0, 60])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Spectrum of filtered noise')
    plt.show()


def za_linenoise_harmonics():
    # load data
    linedata = mr.read_mat('./data/lineNoiseData.mat')
    data = np.squeeze(linedata['data'])
    srate = linedata['srate']

    # time vector
    pnts = len(data)
    time = np.arange(0, pnts) / srate
    time = time.T
    ## narrowband filter to remove line noise

    frex2notch = [50, 150, 250]
    datafilt, pwrfilt, pwr, hz = znl.za_filt_recursive(data, srate, frex2notch)

    ### plot the signal
    plt.subplot(121)
    plt.plot(time, data, 'b', label='Original')
    plt.plot(time, datafilt, 'r', label='Notched')
    plt.xlabel('Time (s)')
    plt.legend()
    # plot power spectrum
    plt.subplot(122)
    plt.plot(hz, pwr, 'b', label='Original')
    plt.plot(hz, pwrfilt, 'r', label='Notched')
    plt.xlim([0, 400])
    plt.ylim([0, 2])
    plt.legend()
    plt.title('Frequency domain')
    plt.show()


def za_3db_rolloff():
    ## create a windowed sinc filter

    # simulation parameters
    srate = 1000
    time = np.arange(-4, 4, 1 / srate)
    pnts = len(time)

    # FFT parameters
    nfft = 10000
    hz = np.linspace(0, srate / 2, int(np.floor(nfft / 2) + 1))

    filtcut = 15
    sincfilt = np.sin(2 * np.pi * filtcut * time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt / np.sum(sincfilt)

    # windowed sinc filter
    sincfiltW = sincfilt * signal.windows.hann(pnts)

    # spectrum of filter
    sincX = 10 * np.log10(np.abs(scipy.fftpack.fft(sincfiltW, n=nfft)) ** 2)
    sincX = sincX[:len(hz)]

    ## create a Butterworth high-pass filter

    # generate filter coefficients (Butterworth)
    filtb, filta = signal.butter(5, filtcut / (srate / 2), btype='lowpass')

    # test impulse response function (IRF)
    impulse = np.zeros(1001)
    impulse[500] = 1
    fimpulse = signal.filtfilt(filtb, filta, impulse)

    # spectrum of filter response
    butterX = 10 * np.log10(np.abs(scipy.fftpack.fft(fimpulse, nfft)) ** 2)
    butterX = butterX[:len(hz)]

    ## plot frequency responses

    plt.plot(hz, sincX, 'g-', label='Sinc')
    plt.plot(hz, butterX, 'k-', label='Butter')

    plotedge = int(np.argmin((hz - filtcut * 3) ** 2))
    plt.xlim([0, filtcut * 3])
    plt.ylim([np.min((butterX[plotedge], sincX[plotedge])), 5])
    # plt.plot([filtcut, filtcut], [-190, 5], 'm-')
    plt.axvline(x=filtcut, ymin=-190, ymax=5, linestyle='-', color='m', label='Cut off Freq = ' + "{:.2f}".format(filtcut))

    sincX3db, butterX3db, sincXoct, butterXoct, sincXrolloff, butterXrolloff = zf3dbr.za_find_3_db(filtcut, hz, sincX,
                                                                                                   butterX)

    # add to the plot
    # plt.plot([hz[sincX3db], hz[sincX3db]], [-180, 5], 'b--')
    plt.axvline(x=hz[sincX3db], ymin=-180, ymax=5, linestyle='--', color='b', label='Sinc 3dB Freq = ' + "{:.2f}".format(hz[sincX3db]))
    plt.axvline(x=hz[butterX3db], ymin=-180, ymax=5, linestyle='--', color='r', label='Butter 3dB Freq = ' + "{:.2f}".format(hz[butterX3db]))
    # plt.plot([hz[sincX3db], hz[sincX3db]], [-180, 5], 'b--')
    # plt.plot([hz[butterX3db], hz[butterX3db]], [-180, 5], 'r--')

    # add to the plot
    plt.axvline(x=hz[sincXoct], ymin=-180, ymax=5, linestyle='-', color='b', label='Sinc 3dB Freq X 2 = ' + "{:.2f}".format(hz[sincXoct]))
    plt.axvline(x=hz[butterXoct], ymin=-180, ymax=5, linestyle='-', color='r', label='Butter 3dB X 2 = ' + "{:.2f}".format(hz[butterXoct]))
    # plt.plot([hz[sincXoct], hz[sincXoct]], [-180, 5], 'b-')
    # plt.plot([hz[butterXoct], hz[butterXoct]], [-180, 5], 'r-')

    # report!
    plt.title('Sinc: %.3f, Butterworth: %.3f' % (sincXrolloff, butterXrolloff))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')

    xax = [hz[0], hz[sincX3db], hz[butterX3db], hz[sincXoct], hz[butterXoct], filtcut]
    xax.sort()
    plt.xticks(xax)
    plt.legend()
    plt.show()
