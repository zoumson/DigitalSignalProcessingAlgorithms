import denoise.statistic.mean as dsm
import denoise.gaussian.smooth as dgs
import denoise.gaussian.spikes as dgk
import denoise.statistic.median as dsd
import denoise.detrend.default as ddd
import denoise.detrend.polynomial as ddp
import denoise.statistic.average as dsa
import denoise.template_projection as dtp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.io as sio
from importlib import resources
import pymatreader as mr
import pkgutil
import scipy.signal
from scipy import *
import copy


def za_mean():
    # create signal
    srate = 1000  # Hz
    time = np.arange(0, 3, 1 / srate)
    n = len(time)
    p = 15  # poles for random interpolation

    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal = ampl + noise

    # initialize filtered signal vector
    filtsig = np.zeros(n)

    # implement the running mean filter
    k = 20  # filter window is actually k*2+1
    filtsig = dsm.za_filt(signal, k)
    # compute window size in ms
    windowsize = 1000 * (k * 2 + 1) / srate

    # plot the noisy and filtered signals
    plt.plot(time, signal, label='original')
    plt.plot(time, filtsig, label='filtered')

    plt.legend()
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.title('Running-mean filter with a k=%d-ms filter' % windowsize)

    plt.show()


def za_gauss_smooth():
    # create signal
    srate = 1000  # Hz
    time = np.arange(0, 3, 1 / srate)
    n = len(time)
    p = 15  # poles for random interpolation

    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(1, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal = ampl + noise
    print(signal[10])
    ## create Gaussian kernel
    # full-width half-maximum: the key Gaussian parameter
    fwhm = 25  # in ms

    # normalized time vector in ms
    k = 100
    ## implement the filtering
    # filtsigG = dgs.za_gs_filt(signal, srate, fwhm, k)
    filtsigG = dgs.za_gs_filt(signal, srate, fwhm, k)
    # plot
    plt.plot(time, signal, 'r', label='Original')
    plt.plot(time, filtsigG, 'k', label='Gaussian-filtered')

    plt.xlabel('Time (s)')
    plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.title('Gaussian smoothing filter')
    plt.show()


def za_gauss_spike():
    ## generate time series of random spikes

    # number of spikes
    n = 300

    # inter-spike intervals (exponential distribution for bursts)
    isi = np.round(np.exp(np.random.randn(n)) * 10)

    # generate spike time series
    spikets = np.zeros(int(sum(isi)))

    for i in range(0, n):
        spikets[int(np.sum(isi[0:i]))] = 1

    # full-width half-maximum: the key Gaussian parameter
    fwhm = 25  # in points
    # normalized time vector in ms
    k = 100
    filtsigG = dgk.za_gk_filt(spikets, fwhm, k)
    # plot the filtered signal (spike probability density)
    plt.plot(spikets, 'b', label='spikes')
    plt.plot(filtsigG, 'r', label='spike p.d.')
    plt.legend()
    plt.title('Spikes and spike probability density')
    plt.show()


def za_median():
    # create signal
    n = 2000
    signal = np.cumsum(np.random.randn(n))

    # proportion of time points to replace with noise
    propnoise = .05

    # find noise points
    noisepnts = np.random.permutation(n)
    noisepnts = noisepnts[0:int(n * propnoise)]

    # generate signal and replace points with noise
    signal[noisepnts] = 50 + np.random.rand(len(noisepnts)) * 100

    # # use hist to pick threshold
    # plt.hist(signal, 100)
    # plt.show()

    # visual-picked threshold
    threshold = 40
    # loop through suprathreshold points and set to median of k
    k = 20  # actual window is k*2+1
    filtsig = dsd.za_md_filt(signal, threshold, k)

    # plot
    plt.plot(range(0, n), signal, range(0, n), filtsig)
    plt.show()


def za_detrend_default():
    # create signal with linear trend imposed
    n = 2000
    signal = np.cumsum(np.random.randn(n)) + np.linspace(-30, 30, n)

    # linear detrending
    detsignal = ddd.za_l_filt(signal)

    # get means
    omean = np.mean(signal)  # original mean
    dmean = np.mean(detsignal)  # detrended mean

    # plot signal and detrended signal
    plt.plot(range(0, n), signal, label='Original, mean=%d' % omean)
    plt.plot(range(0, n), detsignal, label='Detrended, mean=%d' % dmean)

    plt.legend()
    plt.show()


def za_detrend_pol():
    ## generate signal with slow polynomial artifact

    n = 10000
    t = range(n)
    k = 10  # number of poles for random amplitudes

    slowdrift = np.interp(np.linspace(1, k, n), np.arange(0, k), 100 * np.random.randn(k))
    signal = slowdrift + 20 * np.random.randn(n)

    ## fit a 3-order polynomial
    d = 5, 40
    # predicted data is evaluation of polynomial
    residual, yHat, pOrder = ddp.za_p_filt(signal, d)

    # now plot the fit (the function that will be removed)
    plt.plot(t, signal, 'b', label='Original')
    plt.plot(t, yHat, 'r', label='Polyfit')
    plt.plot(t, residual, 'k', label='Filtered signal')
    plt.legend(title='Pol order ' + str(pOrder))
    plt.show()


def za_average():
    ## simulate data

    # create event (derivative of Gaussian)
    k = 100  # duration of event in time points
    event = np.diff(np.exp(-np.linspace(-2, 2, k + 1) ** 2))
    event = event / np.max(event)  # normalize to max=1

    # event onset times
    Nevents = 30
    onsettimes = np.random.permutation(10000 - k)
    onsettimes = onsettimes[0:Nevents]

    # put event into data
    data = np.zeros(10000)
    for ei in range(Nevents):
        data[onsettimes[ei]:onsettimes[ei] + k] = event

    # add noise
    data = data + .5 * np.random.randn(len(data))

    ## extract all events into a matrix

    dataaverage = dsa.za_av_filt(data, Nevents, onsettimes, k)

    plt.plot(range(0, k), dataaverage, label='Averaged')
    plt.plot(range(0, k), event, label='Ground-truth')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Average events')
    plt.show()


def za_tp():
    do_plt_tria_map = True
    do_plt_ori_filt = False
    # load dataset
    matdat = mr.read_mat('./data/templateProjection.mat')

    EEGdat = pd.DataFrame(matdat['EEGdat']).values
    eyedat = pd.DataFrame(matdat['eyedat']).values
    timevec = pd.DataFrame(matdat['timevec'])[0].values

    MN = np.shape(EEGdat)  # matrix sizes
    EEGdat, eyedat, resdat = dtp.za_tp_filt(EEGdat, eyedat)

    def plt_ori_filt():
        # trial averages
        plt.plot(timevec, eyedat, label='EOG')
        plt.plot(timevec, EEGdat, label='EEG')
        plt.plot(timevec, resdat, label='Residual')
        #
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.show()

    if do_plt_ori_filt:
        plt_ori_filt()

    def plt_tria_map():

        # show all trials in a map
        clim = [-1, 1] * 20

        plt.subplot(131)
        plt.imshow(eyedat.T)
        plt.title('EOG')

        plt.subplot(132)
        plt.imshow(EEGdat.T)
        plt.title('Uncorrected EEG')

        plt.subplot(133)
        plt.imshow(resdat.T)
        plt.title('cleaned EEG')

        plt.tight_layout()
        plt.show()

    if do_plt_tria_map:
        plt_tria_map()
