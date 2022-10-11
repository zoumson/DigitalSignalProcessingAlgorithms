import numpy as np
from nptyping import NDArray, Shape, Float, Int
import numpy as np
from scipy import *
import copy


# def za_gs_filt(za_in: NDArray[Shape["*"], Float], srate: Float, fwhm: Float, k: Int) -> NDArray[Shape["*"], Float]:
def za_tp_filt(za_in_1, za_in_2):
    MN = np.shape(za_in_1)  # matrix sizes
    # initialize residual data
    resdat = np.zeros(MN)

    # loop over trials
    for triali in range(MN[1]):
        # build the least-squares model as intercept and EOG from this trial
        X = np.column_stack((np.ones(MN[0]), za_in_2[:, triali]))

        # compute regression coefficients for EEG channel
        b = np.linalg.solve(np.matrix.transpose(X) @ X, np.matrix.transpose(X) @ za_in_1[:, triali])

        # predicted data
        yHat = X @ b

        # new data are the residuals after projecting out the best EKG fit
        resdat[:, triali] = za_in_1[:, triali] - yHat

        return  np.mean(za_in_1, axis=1), np.mean(za_in_2, axis=1), np.mean(resdat, 1)
