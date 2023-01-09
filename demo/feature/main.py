import feature.local_min_max as dfl

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sio
from scipy import signal
import copy
from scipy.interpolate import griddata


def za_local_max_min():
    # signal is sinc plus linear trend
    time_val = np.linspace(-4 * np.pi, 16 * np.pi, 1000)
    sig_val = np.sin(time_val) / time_val + np.linspace(1, -1, len(time_val))
    peeks1_min = dfl.za_find_local_min(sig_val)
    peeks2_min = dfl.za_find_local_min_detrend(sig_val)

    peeks1_max = dfl.za_find_local_max(sig_val)
    peeks2_max = dfl.za_find_local_max_detrend(sig_val)

    plt.plot(time_val, sig_val, label="Signal")
    plt.plot(time_val[peeks1_min], sig_val[peeks1_min], 'ro', label="Local Minima Without Detrending")
    plt.plot(time_val[peeks1_min], sig_val[peeks1_min], 'gs', label="Local Minima With Detrending")

    plt.plot(time_val[peeks1_max], sig_val[peeks1_max], 'ko', label="Local Maxima Without Detrending")
    plt.plot(time_val[peeks1_max], sig_val[peeks1_max], 'ys', label="Local Maxima With Detrending")
    plt.legend()
    plt.title("Local Critical Points")
    plt.show()
