import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy
import scipy

# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_l_filt(za_in):
    # linear detrending
    detsignal = scipy.signal.detrend(za_in)
    return detsignal



