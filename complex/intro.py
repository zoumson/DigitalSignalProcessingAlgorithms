import numpy as np
import cmath
import math
import matplotlib.pyplot as plt

## the imaginary operator

# Python differs from MATLAB here:
1j
cmath.sqrt(-1)  # note the cmath module

def za_comp_create(r, i):
    ## creating complex numbers
    z = 4 + 3j# need to know imag
    z = r + i * 1j
    z = r + i * cmath.sqrt(-1)
    z = complex(r, i)
    return  z