import wavelet.type as zwt

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy
import pymatreader as mr
import pandas as pd

def za_morlet_filt():
    ## general simulation parameters

    fs = 1024
    npnts = fs * 5  # 5 seconds
    # centered time vector
    timevec = np.arange(0, npnts) / fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))
    # parameters
    freq = 4  # peak frequency
    fwhm = .5  # full-width at half-maximum in seconds

    MorletWavelet, MorletWaveletPow = zwt.za_morlet(npnts, freq, fwhm, timevec)

    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec, MorletWavelet, 'k')
    plt.xlabel('Time (sec.)')
    plt.title('Morlet wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz, MorletWaveletPow[:len(hz)], 'k')
    plt.xlim([0, freq * 3])
    plt.xlabel('Frequency (Hz)')
    plt.title('Morlet wavelet in frequency domain')
    plt.show()


def za_haar_filt():
    ## general simulation parameters

    fs = 1024
    npnts = fs * 5  # 5 seconds
    # centered time vector
    timevec = np.arange(0, npnts) / fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))
    # parameters
    freq = 4  # peak frequency
    fwhm = .5  # full-width at half-maximum in seconds

    HaarWavelet, HaarWaveletPow = zwt.za_haar(npnts, fs, timevec)

    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec, HaarWavelet, 'k')
    plt.xlabel('Time (sec.)')
    plt.title('Haar wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz, HaarWaveletPow[:len(hz)], 'k')
    plt.xlim([0, freq * 3])
    plt.xlabel('Frequency (Hz)')
    plt.title('Haar wavelet in frequency domain')
    plt.show()


def za_hat_filt():
    ## general simulation parameters

    fs = 1024
    npnts = fs * 5  # 5 seconds
    # centered time vector
    timevec = np.arange(0, npnts) / fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

    # parameters
    freq = 4  # peak frequency
    fwhm = .5  # full-width at half-maximum in seconds
    s = .4
    MexicanWavelet, MexicanWaveletPow = zwt.za_hat(npnts, s, timevec)

    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec, MexicanWavelet, 'k')
    plt.xlabel('Time (sec.)')
    plt.title('Mexican wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz, MexicanWaveletPow[:len(hz)], 'k')
    plt.xlim([0, freq * 3])
    plt.xlabel('Frequency (Hz)')
    plt.title('Mexican wavelet in frequency domain')
    plt.show()


def za_dog_filt():
    ## general simulation parameters

    fs = 1024
    npnts = fs * 5  # 5 seconds
    # centered time vector
    timevec = np.arange(0, npnts) / fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

    # parameters
    freq = 4  # peak frequency
    # define sigmas
    sPos = .1
    sNeg = .5

    DoG, DoGPow = zwt.za_dog(npnts, sPos, sNeg, timevec)

    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec, DoG, 'k')
    plt.xlabel('Time (sec.)')
    plt.title('DoG wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz, DoGPow[:len(hz)], 'k')
    plt.xlim([0, freq * 3])
    plt.xlabel('Frequency (Hz)')
    plt.title('DoG wavelet in frequency domain')
    plt.show()


def za_wavelet_filt():
    ### create wavelets
    ## general simulation parameters

    fs = 1024
    npnts = fs * 5  # 5 seconds

    # centered time vector
    timevec = np.arange(0, npnts) / fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

    # parameters
    freq = 4  # peak frequency
    fwhm = .5  # full-width at half-maximum in seconds
    ## Morlet wavelet
    MorletWavelet, _ = zwt.za_morlet(npnts, freq, fwhm, timevec)
    ## Haar wavelet
    HaarWavelet, _ = zwt.za_haar(npnts, fs, timevec)

    ## Mexican hat wavelet
    s = .4
    MexicanWavelet, _ = zwt.za_hat(npnts, s, timevec)

    ## convolve with random signal

    # signal
    signal1 = scipy.signal.detrend(np.cumsum(np.random.randn(npnts)))

    # convolve signal with different wavelets
    morewav = np.convolve(signal1, MorletWavelet, 'same')
    haarwav = np.convolve(signal1, HaarWavelet, 'same')
    mexiwav = np.convolve(signal1, MexicanWavelet, 'same')

    # amplitude spectra
    morewaveAmp = np.abs(scipy.fftpack.fft(morewav) / npnts)
    haarwaveAmp = np.abs(scipy.fftpack.fft(haarwav) / npnts)
    mexiwaveAmp = np.abs(scipy.fftpack.fft(mexiwav) / npnts)

    ### plotting
    # the signal
    plt.plot(timevec, signal1, 'k')
    plt.title('Signal')
    plt.xlabel('Time (s)')
    plt.show()

    # the convolved signals
    plt.subplot(211)
    plt.plot(timevec, morewav, label='Morlet')
    plt.plot(timevec, haarwav, label='Haar')
    plt.plot(timevec, mexiwav, label='Mexican')
    plt.title('Time domain')
    plt.legend()

    # spectra of convolved signals
    plt.subplot(212)
    plt.plot(hz, morewaveAmp[:len(hz)], label='Morlet')
    plt.plot(hz, haarwaveAmp[:len(hz)], label='Haar')
    plt.plot(hz, mexiwaveAmp[:len(hz)], label='Mexican')
    plt.yscale('log')
    plt.xlim([0, 40])
    plt.title('Frequency domain')
    plt.legend()
    plt.xlabel('Frequency (Hz.)')

    plt.show()


def za_wavelet_narrow_band_filt():
    # simulation parameters
    srate = 4352  # hz
    npnts = 8425
    time = np.arange(0, npnts) / srate
    hz = np.linspace(0, srate / 2, int(np.floor(npnts / 2) + 1))

    # pure noise signal
    signal1 = np.exp(.5 * np.random.randn(npnts))

    fig = plt.figure()
    fig.suptitle('Original Signal')
    # let's see what it looks like
    ax = plt.subplot(211)
    ax.plot(time, signal1, 'b')
    ax.set_xlabel('Time (s)')
    ax.set_title('Time Domain')
    # in the frequency domain
    signalX = 2 * np.abs(scipy.fftpack.fft(signal1))
    ax = plt.subplot(212)
    ax.plot(hz, signalX[:len(hz)], 'r')
    ax.set_xlim([1, srate / 6])
    ax.set_ylim([0, 300])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Frequency Domain')
    plt.show()

    ## create and inspect the Morlet wavelet

    # wavelet parameters
    ffreq = 34  # filter frequency in Hz
    fwhm = .12  # full-width at half-maximum in seconds
    wavtime = np.arange(-3, 3, 1 / srate)  # wavelet time vector (same sampling rate as signal!)
    # amplitude spectrum of wavelet
    # (note that the wavelet needs its own hz because different length)
    wavehz = np.linspace(0, srate / 2, int(np.floor(len(wavtime) / 2) + 1))
    morwav, morwavX = zwt.za_morlet(npnts, ffreq, fwhm, wavtime)

    fig = plt.figure()
    fig.suptitle('Morlet Wavelet(cos*exp) as narrow band filter kernel')
    # plot it!
    ax = plt.subplot(211)
    ax.plot(wavtime, morwav, 'k')
    ax.set_xlim([-.5, .5])
    ax.set_xlabel('Time (sec.)')
    ax.set_title('Time domain')

    ax = plt.subplot(212)
    ax.plot(wavehz, morwavX[:len(wavehz)], 'k')
    ax.set_xlim([0, ffreq * 2])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Frequency domain')
    plt.show()

    ## now for convolution

    convres = scipy.signal.convolve(signal1, morwav, 'same')

    fig = plt.figure()
    fig.suptitle('Built-in convolve filtering')
    # show in the time domain
    ax = plt.subplot(211)
    ax.set_title("Time domain")
    # plt.title('Built-in convolve')
    ax.plot(time, convres, 'r')

    # and in the frequency domain
    ax = plt.subplot(212)
    convresX = 2 * np.abs(scipy.fftpack.fft(convres))
    ax.plot(hz, convresX[:len(hz)], 'r')
    ax.set_title("Frequency domain")
    plt.show()
    ### Time-domain wavelet normalization is... annoying and difficult.
    ### Let's do it in the frequency domain

    ### "manual" convolution

    nConv = npnts + len(wavtime) - 1
    halfw = int(np.floor(len(wavtime) / 2))

    # spectrum of wavelet
    morwavX = scipy.fftpack.fft(morwav, nConv)

    # now normalize in the frequency domain
    ## note: ensure we're taking the magnitude of the peak;
    #  I didn't explain this in the video but it ensures normalization by
    #  the magnitude and not the complex value.
    morwavX = morwavX / np.abs(np.max(morwavX))
    # also equivalent:
    morwavX = (np.abs(morwavX) / max(np.abs(morwavX))) * np.exp(1j * np.angle(morwavX))

    # now for the rest of convolution
    convres = scipy.fftpack.ifft(morwavX * scipy.fftpack.fft(signal1, nConv))
    convres = np.real(convres[halfw:-halfw + 1])

    fig = plt.figure()
    fig.suptitle('Manual convolve filtering')
    # time domain
    ax = plt.subplot(211)
    ax.plot(time, signal1, 'k', label='original')
    ax.plot(time, convres, 'b', label='filtered, norm.')
    ax.legend()
    ax.set_xlabel('Time')
    # ax.xlabel('Time')
    ax.set_title('Time domain')
    # plt.show()

    # frequency domain
    convresX = 2 * np.abs(scipy.fftpack.fft(convres))

    ax = plt.subplot(212)
    ax.plot(hz, signalX[:len(hz)], 'k', label='original')
    ax.plot(hz, convresX[:len(hz)], 'b', label='filtered, norm.')
    ax.set_ylim([0, 300])
    ax.set_xlim([0, 90])
    ax.set_title('Frequency domain')
    ax.legend()
    plt.show()

    ## to preserve DC offset, compute and add back

    convres = convres + np.mean(signal1)

    plt.plot(time, signal1, 'k', label='original')
    plt.plot(time, convres, 'm', label='filtered, norm.')
    plt.legend()
    plt.xlabel('Time')
    plt.title('Preserve DC offset')
    plt.show()


def za_morlet_complx_filt_1():
    # data from http://www.vibrationdata.com/Solomon_Time_History.zip

    equake = np.loadtxt('./data/Solomon_Time_History.txt')

    # more convenient

    times = equake[:, 0]
    equake = equake[:, 1]
    srate = np.round(1 / np.mean(np.diff(times)))

    winsize = srate * 60 * 10  # window size of 10 minutes
    f, welchpow = scipy.signal.welch(equake, fs=srate, window=np.hanning(winsize), nperseg=winsize,
                                     noverlap=winsize / 4)

    ## plot the signal

    fig = plt.figure()
    fig.suptitle('Original Signal')
    # time domain
    ax = plt.subplot(211)
    ax.plot(times / 60 / 60, equake)
    ax.set_xlim([times[0] / 60 / 60, times[-1] / 60 / 60])
    ax.set_xlabel('Time (hours)')
    ax.set_title('Time domain')

    # frequency domain using pwelch
    ax = plt.subplot(212)

    ax.semilogy(f, welchpow)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Power')
    ax.set_ylim([10e-11, 10e-6])
    ax.set_title('Frequency domain')
    plt.show()

    ## setup time-frequency analysis

    # parameters (in Hz)
    numFrex = 40
    minFreq = 2
    maxFreq = srate / 2
    npntsTF = 1000  # this one's in points

    # frequencies in Hz
    frex = np.linspace(minFreq, maxFreq, numFrex)

    # wavelet widths (FWHM in seconds)
    fwhms = np.linspace(5, 15, numFrex)

    # time points to save for plotting
    tidx = np.arange(1, len(times), npntsTF)

    # setup wavelet and convolution parameters
    wavet = np.arange(-10, 10, 1 / srate)
    halfw = int(np.floor(len(wavet) / 2))
    nConv = len(times) + len(wavet) - 1

    # create family of Morlet wavelets
    cmw = np.zeros((len(wavet), numFrex), dtype=complex)
    # loop over frequencies and create wavelets

    for fi in range(0, numFrex):
        # cmw[:, fi] = np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-(4 * np.log(2) * wavet ** 2) / fwhms[fi] ** 2)
        cmw[:, fi] = zwt.za_morlet_complex(frex[fi], fwhms[fi], wavet)

    # plot them
    plt.pcolormesh(wavet, frex, np.abs(cmw).T, vmin=0, vmax=1)
    plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
    plt.title('Complex Morlet as kernel for time-frequency analysis')
    plt.show()

    ## run convolution

    # initialize time-frequency matrix
    tf = np.zeros((len(frex), len(tidx)))
    tfN = np.zeros((len(frex), len(tidx)))

    # baseline time window for normalization
    basetidx = [0, 0]
    basetidx[0] = np.argmin((times - -1000) ** 2)
    basetidx[1] = np.argmin(times ** 2)
    basepow = np.zeros(numFrex)

    # spectrum of data
    dataX = scipy.fftpack.fft(equake, nConv)

    # loop over frequencies for convolution
    for fi in range(0, numFrex):
        # create wavelet
        waveX = scipy.fftpack.fft(cmw[:, fi], nConv)

        ## note: ensure we're taking the magnitude of the peak;
        #  I didn't explain this in the video but it ensures normalization by
        #  the magnitude and not the complex value.
        waveX = waveX / np.abs(np.max(waveX))

        # convolve
        as1 = scipy.fftpack.ifft(waveX * dataX)
        # trim
        as1 = as1[halfw:-halfw]

        # power time course at this frequency
        powts = np.abs(as1) ** 2

        # baseline (pre-quake)
        basepow[fi] = np.mean(powts[range(basetidx[0], basetidx[1])])

        tf[fi, :] = 10 * np.log10(powts[tidx])
        tfN[fi, :] = 10 * np.log10(powts[tidx] / basepow[fi])

    ## show time-frequency maps

    # "raw" power
    plt.subplot(211)
    plt.pcolormesh(times[tidx], frex, tf, vmin=-150, vmax=-70)
    plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
    plt.title('"Raw" time-frequency power')

    # pre-quake normalized power
    plt.subplot(212)
    plt.pcolormesh(times[tidx], frex, tfN, vmin=-15, vmax=15)
    plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
    plt.title('"Raw" time-frequency power')
    plt.show()

    ## normalized and non-normalized power
    plt.subplot(211)
    plt.plot(frex, np.mean(tf, axis=1), 'ks-')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (10log_{10})')
    plt.title('Raw power')

    plt.subplot(212)
    plt.plot(frex, np.mean(tfN, axis=1), 'ks-')
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (norm.)')
    plt.title('Pre-quake normalized power')
    plt.show()


def za_morlet_complx_filt_2():
    # load in data
    braindat = mr.read_mat('./data/data4TF.mat')

    timevec = pd.DataFrame(braindat['timevec'])[0].values
    data = pd.DataFrame(braindat['data'])[0].values
    srate = braindat['srate']

    # plot the signal
    plt.plot(timevec, data)
    plt.xlabel('Time (s)'), plt.ylabel('Voltage (\muV)')
    plt.title('Time-domain signal')
    plt.show()

    ## create complex Morlet wavelets

    # wavelet parameters
    nfrex = 50  # 50 frequencies
    frex = np.linspace(8, 70, nfrex)
    fwhm = .2  # full-width at half-maximum in seconds

    # time vector for wavelets
    wavetime = np.arange(-2, 2, 1 / srate)

    # initialize matrices for wavelets
    wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

    # create complex Morlet wavelet family
    for wi in range(0, nfrex):
        # # Gaussian
        # gaussian = np.exp(-(4 * np.log(2) * wavetime ** 2) / fwhm ** 2)
        # gaussian = np.exp(-(4 * np.log(2) * wavetime ** 2) / fwhm ** 2)
        #
        # # complex Morlet wavelet
        # wavelets[wi, :] = np.exp(1j * 2 * np.pi * frex[wi] * wavetime) * gaussian
        wavelets[wi, :] = zwt.za_morlet_complex(frex[wi], fwhm, wavetime)

    # show the wavelets
    plt.plot(wavetime, np.real(wavelets[10, :]), label='Real part')
    plt.plot(wavetime, np.imag(wavelets[10, :]), label='Imag part')
    plt.xlabel('Time')
    plt.xlim([-.5, .5])
    plt.legend()
    plt.show()

    plt.pcolormesh(wavetime, frex, np.real(wavelets))
    plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
    plt.title('Real part of wavelets')
    plt.xlim([-.5, .5])
    plt.show()

    ## run convolution using spectral multiplication

    # convolution parameters
    nconv = len(timevec) + len(wavetime) - 1  # M+N-1
    halfk = int(np.floor(len(wavetime) / 2))

    # Fourier spectrum of the signal
    dataX = scipy.fftpack.fft(data, nconv)

    # initialize time-frequency matrix
    tf = np.zeros((nfrex, len(timevec)))

    # convolution per frequency
    for fi in range(0, nfrex):
        # FFT of the wavelet
        waveX = scipy.fftpack.fft(wavelets[fi, :], nconv)
        # amplitude-normalize the wavelet
        waveX = waveX / np.abs(np.max(waveX))

        # convolution
        convres = scipy.fftpack.ifft(waveX * dataX)
        # trim the "wings"
        convres = convres[halfk - 1:-halfk]

        # extract power from complex signal
        tf[fi, :] = np.abs(convres) ** 2

    ## plot the results

    plt.pcolormesh(timevec, frex, tf, vmin=0, vmax=1e3)
    plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency power')
    plt.show()