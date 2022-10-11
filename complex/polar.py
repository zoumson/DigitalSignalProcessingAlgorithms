import numpy as np
import cmath
import math
import matplotlib.pyplot as plt


def za_comp_mag_phaz(r1, i1):
    # create a complex number
    z = complex(r1, i1)

    # magnitude of the number (distance to origin)
    magZ1 = np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2)
    magZ2 = np.abs(z)

    # angle of the line relative to positive real axis
    angZ1 = math.atan2(np.imag(z), np.real(z))
    angZ2 = np.angle(z)

    return z, angZ1, magZ1
