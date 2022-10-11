import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy
import scipy

# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_p_filt(za_in, k):
    ## fit a k-order polynomial
    n = len(za_in)
    t = range(n)

    ## Bayes information criterion to find optimal order

    # possible orders
    # orders = range(5, 40)
    # print(k[0])
    # print(k[1])
    orders = range(k[0], k[1])

    # sum of squared errors (sse is reserved!)
    sse1 = np.zeros(len(orders))

    # loop through orders
    for ri in range(len(orders)):
        # compute polynomial (fitting time series)
        yHat = np.polyval(np.polyfit(t, za_in, orders[ri]), t)

        # compute fit of model to data (sum of squared errors)
        sse1[ri] = np.sum((yHat - za_in) ** 2) / n

    # Bayes information criterion
    bic = n * np.log(sse1) + orders * np.log(n)

    # best parameter has lowest BIC
    bestP = min(bic)
    idx = np.argmin(bic)
    ## now repeat filter for best (smallest) BIC

    # polynomial fit
    polycoefs = np.polyfit(t, za_in, orders[idx])

    # estimated data based on the coefficients
    yHat = np.polyval(polycoefs, t)

    # filtered signal is residual
    filtsig = za_in - yHat
    return filtsig, yHat, orders[idx]


