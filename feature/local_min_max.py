import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sio
from scipy import signal
import copy
from scipy.interpolate import griddata


def za_find_local_min(sig):
    # find local maxima and plot
    peeks_idx = np.squeeze(np.where(np.diff(np.sign(np.diff(sig))) > 0)) + 1

    return peeks_idx


def za_find_local_max(sig):
    # find local maxima and plot
    peeks_idx = np.squeeze(np.where(np.diff(np.sign(np.diff(sig))) < 0)) + 1

    return peeks_idx


def za_find_local_min_detrend(sig):
    # find local maxima and plot
    peeks_idx = np.squeeze(np.where(np.diff(np.sign(np.diff(scipy.signal.detrend(sig)))) > 0)) + 1

    return peeks_idx


def za_find_local_max_detrend(sig):
    # find local maxima and plot
    peeks_idx = np.squeeze(np.where(np.diff(np.sign(np.diff(scipy.signal.detrend(sig)))) < 0)) + 1

    return peeks_idx
