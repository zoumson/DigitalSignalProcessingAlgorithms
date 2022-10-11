import numpy as np
import cmath
import math
import matplotlib.pyplot as plt



def za_comp_div(r1, r2, i1, i2):
    # create two complex numbers
    a = complex(r1, i1)
    b = complex(r2, i2)
    # let Python do the hard work
    z1 = a / b

    # the "manual" way
    z2 = (a * np.conj(b)) / (b * np.conj(b))

    return z1