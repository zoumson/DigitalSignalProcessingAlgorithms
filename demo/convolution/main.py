import convolution.time_usage as ctu
import convolution.theorem as ctt
import convolution.kernel_type as ckt









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




# VIDEO: Convolution with frequency-domain Gaussian (narrowband filter)
def za_tim_freq_2():
    ## create signal
    srate = 1000  # Hz
    time = np.arange(0, 3, 1 / srate)
    n = len(time)
    p = 15  # poles for random interpolation

    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal1 = ampl + noise

    # subtract mean to eliminate DC
    signal1 = signal1 - np.mean(signal1)

    ## create Gaussian spectral shape
    # Gaussian parameters (in Hz)
    peakf = 11
    fwhm = 5.2

    # vector of frequencies
    hz = np.linspace(0, srate, n)

    # frequency-domain Gaussian
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
    x = hz - peakf  # shifted frequencies
    fx = np.exp(-.5 * (x / s) ** 2)  # gaussian

    ## now for convolution

    # FFTs
    dataX = scipy.fftpack.fft(signal1)

    # IFFT
    convres = 2 * np.real(scipy.fftpack.ifft(dataX * fx))

    # frequencies vector
    hz = np.linspace(0, srate, n)

    ### time-domain plot

    # lines
    plt.plot(time, signal1, 'r', label='Signal')
    plt.plot(time, convres, 'k', label='Smoothed')
    plt.xlabel('Time (s)'), plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.title('Narrowband filter')
    plt.show()

    ### frequency-domain plot

    # plot Gaussian kernel
    plt.plot(hz, fx, 'k')
    plt.xlim([0, 30])
    plt.ylabel('Gain')
    plt.title('Frequency-domain Gaussian')
    plt.show()

    # raw and filtered data spectra
    plt.plot(hz, np.abs(dataX) ** 2, 'rs-', label='Signal')
    plt.plot(hz, np.abs(dataX * fx) ** 2, 'bo-', label='Conv. result')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.title('Frequency domain')
    plt.xlim([0, 25])
    plt.ylim([0, 1e6])
    plt.show()

def za_tim_gauss():
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

    ## create the Gaussian kernel
    # full-width half-maximum: the key Gaussian parameter
    fwhm = 25  # in ms

    # normalized time vector in ms
    k = 100

    fSigTConv, fSigTXNoEdge, kernX, sigX, fSigX, kernXP, sigXP, fSigXP, hz = ckt.za_gauss_time_smooth_filt(fwhm, k, srate, signal1)

    ### time-domain plot

    # lines
    plt.plot(tim, signal1, 'r', label='Signal')
    plt.plot(tim, fSigTConv, 'k*', label='Filtered Signal(Time-domain)')
    plt.plot(tim, fSigTXNoEdge, 'bo', label='Filtered Signal(Spectral mult)')
    plt.xlabel('Time (s)')
    plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.show()

    ### frequency-domain plot

    # plot Gaussian kernel
    plt.plot(hz, kernXP)
    plt.xlim([0, 30])
    plt.ylabel('Gain'), plt.xlabel('Frequency (Hz)')
    plt.title('Power spectrum of Gaussian')
    plt.show()

    # raw and filtered data spectra
    plt.plot(hz, sigXP, 'rs-', label='Signal')
    plt.plot(hz, fSigXP, 'bo-', label='Conv. result')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.title('Frequency domain')
    plt.xlim([0, 25])
    plt.ylim([0, 400000])
    plt.show()

def za_freq_gauss():
    ## create signal
    srate = 1000  # Hz
    tim = np.arange(0, 3, 1 / srate)
    n = len(tim)
    p = 15  # poles for random interpolation

    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal1 = ampl + noise

    # subtract mean to eliminate DC
    signal1 = signal1 - np.mean(signal1)

    ## create Gaussian spectral shape
    # Gaussian parameters (in Hz)
    peakf = 11
    fwhm = 5.2

    sigX, kernX, fSigX, sigXP, fSigXP, fSig, hz = ckt.za_gauss_freq_narrow_band_filt(peakf, fwhm, srate, signal1)

    ### time-domain plot

    # lines
    plt.plot(tim, signal1, 'r', label='Signal')
    plt.plot(tim, fSig, 'k', label='Smoothed')
    plt.xlabel('Time (s)'), plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.title('Narrowband filter')
    plt.show()

    ### frequency-domain plot

    # plot Gaussian kernel
    plt.plot(hz, kernX, 'k')
    plt.xlim([0, 30])
    plt.ylabel('Gain')
    plt.title('Frequency-domain Gaussian')
    plt.show()

    # raw and filtered data spectra
    plt.plot(hz, sigXP, 'rs-', label='Signal')
    plt.plot(hz, fSigXP, 'bo-', label='Conv. result')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.title('Frequency domain')
    plt.xlim([0, 25])
    plt.ylim([0, 1e6])
    plt.show()


def za_freq_plank():
    ## create the signal

    srate = 1000  # Hz
    tim = np.arange(0, 3, 1 / srate)
    n = len(tim)
    p = 15  # poles for random interpolation

    # noise level, measured in standard deviations
    noiseamp = 5

    # amplitude modulator and noise level
    ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
    noise = noiseamp * np.random.randn(n)
    signal1 = ampl + noise

    # subtract mean to eliminate DC
    signal1 = signal1 - np.mean(signal1)

    plt.plot(tim, signal1)
    plt.show()

    ## create Planck spectral shape

    # frequencies
    hz = np.linspace(0, srate, n)

    # edge decay, must be between 0 and .5
    eta = .15

    # spectral parameters
    fwhm = 13
    peakf = 20
    fSig, kernX, sigXP, fSigXP, hz = ckt.za_plank_band_pass_filt(srate, signal1, fwhm, peakf, eta)

    ### time-domain plots

    # lines
    plt.plot(tim, signal1, 'r', label='Signal')
    plt.plot(tim, fSig, 'k', label='Smoothed')
    plt.xlabel('Time (s)'), plt.ylabel('amp. (a.u.)')
    plt.legend()
    plt.title('Narrowband filter')
    plt.show()

    ### frequency-domain plot

    # plot Gaussian kernel
    plt.plot(hz, kernX, 'k')
    plt.xlim([0, peakf * 2])
    plt.ylabel('Gain')
    plt.title('Frequency-domain Planck taper')
    plt.show()

    # raw and filtered data spectra
    plt.plot(hz, sigXP, 'rs-', label='Signal')
    plt.plot(hz, fSigXP, 'bo-', label='Conv. result')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.title('Frequency domain')
    plt.xlim([0, peakf * 2])
    plt.ylim([0, 1e6])
    plt.show()













