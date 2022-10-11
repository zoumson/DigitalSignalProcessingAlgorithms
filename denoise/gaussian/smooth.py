import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy


# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_gs_filt(za_in, srate, fwhm, k):
    ## create Gaussian kernel
    # full-width half-maximum: the key Gaussian parameter
    # fwhm = 25  # in ms

    # normalized time vector in ms
    # k = 100
    gtime = 1000 * np.arange(-k, k) / srate

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)

    # compute empirical FWHM
    pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
    prePeakHalf = np.argmin((gauswin - .5) ** 2)

    empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]

    # show the Gaussian
    # plt.plot(gtime, gauswin, 'ko-')
    # plt.plot([gtime[prePeakHalf], gtime[pstPeakHalf]], [gauswin[prePeakHalf], gauswin[pstPeakHalf]], 'm')

    # then normalize Gaussian to unit energy
    gauswin = gauswin / np.sum(gauswin)
    # title([ 'Gaussian kernel with requeted FWHM ' num2str(fwhm) ' ms (' num2str(empFWHM) ' ms achieved)' ])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Gain')

    # plt.show()
    ## implement the filter
    # initialize filtered signal vector
    n = len(za_in)
    filtsigG = copy.deepcopy(za_in)
    # filtsigG = za_in
    # print(signal)
    # # implement the running mean filter
    for i in range(k + 1, n - k):
        # each point is the weighted average of k surrounding points
        filtsigG[i] = np.sum(za_in[i - k:i + k] * gauswin)

    return filtsigG


