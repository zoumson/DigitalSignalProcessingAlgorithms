import numpy as np
from nptyping import NDArray, Shape, Float, Int


def za_filt(za_in: NDArray[Shape["*"], Float], k: Int) -> NDArray[Shape["*"], Float]:
    n = len(za_in)
    filtsig = np.zeros(n)
    # implement the running mean filter
    # filter window is actually k*2+1
    for i in range(k, n - k):
        # each point is the average of k surrounding points
        filtsig[i] = np.mean(za_in[i - k:i + k])
    return filtsig
