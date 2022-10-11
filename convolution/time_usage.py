import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time


def za_tim_built_in(signal1, kernel, conv_mode):
    # full, same, valid
    conv_sig = np.convolve(signal1, kernel, conv_mode)
    return conv_sig


def za_tim_manual(signal1, kernel):
    # conv_mode is same
    # convolution sizes
    nSign = len(signal1)
    nKern = len(kernel)
    nConv = nSign + nKern - 1
    ## convolution in animation

    half_kern = int(np.floor(nKern / 2))

    # flipped version of kernel
    kflip = kernel[::-1]  # -np.mean(kernel)

    # zero-padded data for convolution
    dat4conv = np.concatenate((np.zeros(half_kern), signal1, np.zeros(half_kern)), axis=0)

    # initialize convolution output
    conv_res = np.zeros(nConv)

    # run convolution
    for ti in range(half_kern, nConv - half_kern):
        # get a chunk of data
        tempdata = dat4conv[ti - half_kern:ti + half_kern + 1]
        # compute dot product (don't forget to flip the kernel backwards!)
        conv_res[ti] = np.sum(tempdata * kflip)
    # cut off edges
    conv_res = conv_res[half_kern:-half_kern]
    return conv_res
