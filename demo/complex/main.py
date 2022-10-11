import complex.intro as zci
import complex.add_sub as zcas
import complex.mult as zcm
import complex.conj as zcc
import complex.division as zcd
import complex.polar as zcp

import numpy as np
import cmath
import math
import matplotlib.pyplot as plt


def za_comp_creat():
    r = 4
    i = 3
    z = zci.za_comp_create(r, i)
    print('Real part is ' + str(np.real(z)) + ' and imaginary part is ' + str(np.imag(z)) + '.')
    ## plotting a complex number

    plt.axhline(y=3, color='b', linestyle='--')
    plt.axvline(x=4, color='b', linestyle='--')
    plt.plot(np.real(z), np.imag(z), 'rs')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xticks([-5, r, 5])
    plt.yticks([-5, i, 5])
    plt.plot([-5, 5], [0, 0], 'k')
    plt.plot([0, 0], [-5, 5], 'k')
    plt.xlabel('real axis')
    plt.ylabel('imag axis')
    # plt.grid(axis='y')

    plt.show()


def za_comp_add_su():
    r1, r2 = 4, 3
    i1, i2 = 5, 2

    z1, z2 = zcas.za_comp_add_sub(r1, r2, i1, i2)

    print("Adding:\n" + "First complex: real = " + str(r1) + ", imag = " + str(i1) + "\nSecond complex: real = " + str(
        r2) + ", imag = " + str(i2) + "\nResult = " + str(z1))
    print("\n\nSubstracting:\n" + "First complex: real = " + str(r1) + ", imag = " + str(
        i1) + "\nSecond complex: real = " + str(
        r2) + ", imag = " + str(i2) + "\nResult = " + str(z2))


def za_comp_mul():
    r1, r2 = 4, 3
    i1, i2 = 5, 2

    z1 = zcm.za_comp_mult(r1, r2, i1, i2)

    print("Multiplying:\n" + "First complex: real = " + str(r1) + ", imag = " + str(
        i1) + "\nSecond complex: real = " + str(
        r2) + ", imag = " + str(i2) + "\nResult = " + str(z1))


def za_comp_con():
    r1 = 4
    i1, i2 = 5, 2

    z1, magSquared = zcc.za_comp_conj(r1, i1)

    print("Conjugate:\n" + "Complex Number: real = " + str(r1) + ", imag = " + str(
        i1) + "\nResult Conjugate = " + str(z1) + "\nResult Magnitude Squared = " + str(magSquared))


def za_comp_di():
    r1, r2 = 4, 7
    i1, i2 = -5, 8

    z1 = zcd.za_comp_div(r1, r2, i1, i2)

    print("Division:\n" + "First complex: real = " + str(r1) + ", imag = " + str(
        i1) + "\nSecond complex: real = " + str(
        r2) + ", imag = " + str(i2) + "\nResult = " + str(z1))


def za_comp_polar():
    r1 = 4
    i1 = 3

    z, angZ1, magZ1 = zcp.za_comp_mag_phaz(r1, i1)

    # plot the complex number
    plt.plot(np.real(z), np.imag(z), 'ks')

    # make plot look nicer
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.plot([-5, 5], [0, 0], 'k')
    plt.plot([0, 0], [-5, 5], 'k')
    plt.xlabel('real axis')
    plt.ylabel('imag axis')
    plt.show()

    # draw a line using polar notation
    plt.polar([0, angZ1], [0, magZ1], 'r')
    plt.show()
