import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy


# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_md_filt(za_in, threshold, k):
    # find data values above the threshold
    suprathresh = np.where(za_in > threshold)[0]
    n = len(za_in)
    # initialize filtered signal
    filtsig = copy.deepcopy(za_in)

    # loop through suprathreshold points and set to median of k
    # k = 20  # actual window is k*2+1
    for ti in range(len(suprathresh)):
        # lower and upper bounds
        lowbnd = np.max((0, suprathresh[ti] - k))
        uppbnd = np.min((suprathresh[ti] + k, n + 1))

        # compute median of surrounding points
        filtsig[suprathresh[ti]] = np.median(za_in[lowbnd:uppbnd])

    return filtsig



