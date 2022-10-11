import numpy as np
import cmath
import math
import matplotlib.pyplot as plt



def za_comp_mult(r1, r2, i1, i2):
    # create two complex numbers
    a = complex(r1, i1)
    b = complex(r2, i2)

    # let Python do the hard work
    z1 = a * b

    # the "manual" way
    # the intuitive-but-WRONG way
    z2 = complex(np.real(a) * np.real(b), np.imag(a) * np.imag(b))
    # the less-intuitive-but-CORRECT way
    ar = np.real(a)
    ai = np.imag(a)
    br = np.real(b)
    bi = np.imag(b)

    z3 = (ar + 1j * ai) * (br + 1j * bi)
    z4 = (ar * br) + (ar * (1j * bi)) + ((1j * ai) * br) + ((1j * ai) * (1j * bi))


    return z1