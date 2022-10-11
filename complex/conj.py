import numpy as np
import cmath
import math
import matplotlib.pyplot as plt



def za_comp_conj(r1, i1):
    # create two complex numbers
    a = complex(r1, i1)
    # let Python do the hard work
    z1 = np.conj(a)

    # the "manual" way
    z2 = complex(np.real(a), -np.imag(a))

    ## magnitude squared of a complex number
    amag1 = a * np.conj(a)  # note that Python keeps this as type==complex, although the imaginary part is zero
    amag2 = np.real(a) ** 2 + np.imag(a) ** 2
    amag3 = np.abs(a) ** 2

    return z1, amag3