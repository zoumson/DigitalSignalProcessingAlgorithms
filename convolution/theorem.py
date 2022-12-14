import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time



def za_cov_time_freq(signal1, kernel):
    ## time-domain convolution

    # convolution sizes
    nSign = len(signal1)

    # odd number
    nKern = len(kernel)
    nConv = nSign + nKern - 1

    print(nSign)
    print(nKern)

    half_kern = int(np.floor(nKern / 2))

    # flipped version of kernel
    kflip = kernel[::-1]

    # zero-padded data for convolution
    dat4conv = np.concatenate((np.zeros(half_kern), signal1, np.zeros(half_kern)), axis=0)

    # initialize convolution output
    conv_res = np.zeros(nConv)

    # run convolution
    for ti in range(half_kern, nConv - half_kern):
        # get a chunk of data
        tempdata = dat4conv[ti - half_kern:ti + half_kern + 1]
        # tempdata = dat4conv[ti - half_kern:ti + half_kern]

        # compute dot product (don't forget to flip the kernel backwards!)
        conv_res[ti] = np.sum(tempdata * kflip)

    # cut off edges
    conv_time = conv_res[half_kern + 1:-half_kern]

    ## convolution implemented in the frequency domain

    # spectra of signal and kernel
    signalX = scipy.fftpack.fft(signal1, nConv)
    kernelX = scipy.fftpack.fft(kernel, nConv)

    # element-wise multiply
    sigXkern = signalX * kernelX

    # inverse FFT to get back to the time domain
    conv_resFFT = np.real(scipy.fftpack.ifft(sigXkern))

    # cut off edges
    conv_freq = conv_resFFT[half_kern + 1:-half_kern]

    return conv_time, conv_freq