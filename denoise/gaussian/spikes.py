import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy


# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_gk_filt(za_in, fwhm, k):
    ## create and implement Gaussian window

    # full-width half-maximum: the key Gaussian parameter
    # fwhm = 25  # in points

    # normalized time vector in ms
    # k = 100
    gtime = np.arange(-k, k)

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
    gauswin = gauswin / np.sum(gauswin)

    # initialize filtered signal vector
    filtsigG = np.zeros(len(za_in))

    # implement the weighted running mean filter
    for i in range(k + 1, len(za_in) - k):
        filtsigG[i] = np.sum(za_in[i - k:i + k] * gauswin)

    return filtsigG

