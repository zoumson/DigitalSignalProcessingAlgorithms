import numpy as np
import cmath
import math
import matplotlib.pyplot as plt



def za_comp_add_sub(r1, r2, i1, i2):
    # create two complex numbers
    a = complex(r1, i1)
    b = complex(r2, i2)

    # let Python do the hard work
    z1 = a + b

    # the "manual" way
    z2 = complex(np.real(a) + np.real(b), np.imag(a) + np.imag(b))

    ## subtraction is the same as addition...

    # let MATLAB do the hard work
    z3 = a - b

    # the "manual" way
    z4 = complex(np.real(a) - np.real(b), np.imag(a) - np.imag(b))

    return z1, z3