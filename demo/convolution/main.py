import convolution.time_usage as ctu
import convolution.theorem as ctt









import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time



def za_tim_conv_1():
    ## first example to build intuition

    signal1 = np.concatenate((np.zeros(30), np.ones(2), np.zeros(20), np.ones(30), 2 * np.ones(10), np.zeros(30),
                              -np.ones(10), np.zeros(40)), axis=0)
    kernel = np.exp(-np.linspace(-2, 2, 20) ** 2)
    kernel = kernel / sum(kernel)
    N = len(signal1)
    # full, same, valid
    conv_mode = 'same'
    signal1_conv = ctu.za_tim_built_in(signal1, kernel, conv_mode)
    plt.subplot(311)
    plt.plot(kernel, 'k')
    plt.xlim([0, N])
    plt.title('Kernel')

    plt.subplot(312)
    plt.plot(signal1, 'k')
    plt.xlim([0, N])
    plt.title('Signal')

    plt.subplot(313)
    plt.plot(signal1_conv, 'k')
    plt.xlim([0, N])
    plt.title('Convolution result')

    plt.show()


def za_tim_conv_2():
    ## in a bit more detail

    # signal
    signal1 = np.zeros(20)
    signal1[8:15] = 1

    # convolution kernel
    kernel = [1, .8, .6, .4, .2]

    # convolution sizes
    nSign = len(signal1)
    nKern = len(kernel)
    nConv = nSign + nKern - 1
    # full, same, valid
    conv_mode = 'same'
    signal1_conv = ctu.za_tim_built_in(signal1, kernel, conv_mode)
    # plot the signal
    plt.subplot(311)
    plt.plot(signal1, 'o-')
    plt.xlim([0, nSign])
    plt.title('Signal')

    # plot the kernel
    plt.subplot(312)
    plt.plot(kernel, 'o-')
    plt.xlim([0, nSign])
    plt.title('Kernel')

    # plot the result of convolution
    plt.subplot(313)
    plt.plot(signal1_conv, 'o-')
    plt.xlim([0, nSign])
    plt.title('Result of convolution')
    plt.show()

def za_tim_conv_3():

    # signal
    signal1 = np.zeros(20)
    signal1[8:15] = 1

    # convolution kernel
    kernel = [1, .8, .6, .4, .2]

    # convolution sizes
    nSign = len(signal1)
    nKern = len(kernel)
    nConv = nSign + nKern - 1
    # full, same, valid
    conv_mode = 'same'
    signal1_conv_py = ctu.za_tim_built_in(signal1, kernel, conv_mode)
    signal1_conv_manual = ctu.za_tim_manual(signal1, kernel)

    plt.plot(signal1_conv_manual, 'o-', label='Time-domain manual convolution')
    plt.plot(signal1_conv_py, '-', label='np.convolve()')
    plt.legend()
    plt.show()

def za_tim_freq_1():

    # signal
    signal1 = np.zeros(20)
    signal1[8:15] = 1

    # convolution kernel
    kernel = [1, .8, .6, .4, .2]

    # convolution sizes
    nSign = len(signal1)
    nKern = len(kernel)
    nConv = nSign + nKern - 1
    conv_time, conv_freq = ctt.za_cov_time_freq(signal1, kernel)
    ## plot for comparison

    plt.plot(conv_time, 'o-', label='Time domain')
    plt.plot(conv_freq, '-', label='Freq. domain')
    plt.legend()
    plt.show()


def za_tim_freq_2():
    # simulation parameters
    srate = 1000  # Hz
    tim = np.arange(0, 3, 1 / srate)
    n = len(tim)
    p = 15  # poles for random interpolation

    ## create signal
    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal1 = ampl + noise

    # subtract mean to eliminate DC
    signal1 = signal1 - np.mean(signal1)

    # create the Gaussian kernel
    # full-width half-maximum: the key Gaussian parameter
    fwhm = 25  # in ms

    # kernel has to an odd number for convolution match
    # normalized time vector in ms
    k = 100
    gtime = 1000 * np.arange(-k, k + 1) / srate

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)

    # then normalize Gaussian to unit energy
    gauswin = gauswin / np.sum(gauswin)

    print(len(gauswin))

    conv_time, conv_freq = ctt.za_cov_time_freq(signal1, gauswin)
    ## plot for comparison

    ### time-domain plot

    # lines
    plt.plot(tim, signal1, 'r', label='Signal')
    plt.plot(tim, conv_time, 'k*', label='Time-domain')
    # plt.plot(time, convres, 'bo', label='Spectral mult.')
    plt.xlabel('Time (s)')
    plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.show()













