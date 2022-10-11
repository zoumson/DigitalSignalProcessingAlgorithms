import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


def za_find_3_db(filtcut, hz, sincX, butterX):
    # find -3 dB after filter edge
    filtcut_idx = np.min((hz - filtcut) ** 2)

    sincX3db = np.argmin((sincX - -3) ** 2)
    butterX3db = np.argmin((butterX - -3) ** 2)

    # add to the plot
    # plt.plot([hz[sincX3db], hz[sincX3db]], [-180, 5], 'b--')
    # plt.plot([hz[butterX3db], hz[butterX3db]], [-180, 5], 'r--')

    # find double the frequency
    sincXoct = np.argmin((hz - hz[sincX3db] * 2) ** 2)
    butterXoct = np.argmin((hz - hz[butterX3db] * 2) ** 2)

    # add to the plot
    # plt.plot([hz[sincXoct], hz[sincXoct]], [-180, 5], 'b--')
    # plt.plot([hz[butterXoct], hz[butterXoct]], [-180, 5], 'r--')

    # find attenuation from that point to double its frequency
    sincXatten = sincX[sincX3db * 2]
    butterXatten = butterX[butterX3db * 2]

    sincXrolloff = (sincX[sincX3db] - sincX[sincXoct]) / (hz[sincXoct] - hz[sincX3db])
    butterXrolloff = (butterX[butterX3db] - butterX[butterXoct]) / (hz[butterXoct] - hz[butterX3db])

    return sincX3db, butterX3db, sincXoct, butterXoct, sincXrolloff, butterXrolloff
