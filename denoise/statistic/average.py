import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy


# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_av_filt(za_in, Nevents, onsettimes, k):
    datamatrix = np.zeros((Nevents, k))

    for ei in range(Nevents):
        datamatrix[ei, :] = za_in[onsettimes[ei]:onsettimes[ei] + k]

    return np.mean(datamatrix,axis=0)