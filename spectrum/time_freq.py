import numpy as np
import matplotlib.pyplot as plt
# import scipy.io as sio
import scipy.signal
from scipy import *
import copy
import pymatreader as mr
import pandas as pd
# import sounddevice as sd
from time import sleep
import scipy.io.wavfile


def za_t_f(sig, fs):
    frex, tim, pwr = scipy.signal.spectrogram(sig, fs)
    return frex, tim, pwr