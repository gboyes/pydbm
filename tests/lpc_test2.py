import os
import numpy as np
import pydbm.dictionary
import pydbm.meta
import pydbm.data
import pydbm.atom
import pydbm.utils

import matplotlib.pyplot as plt
import audiolab

import scipy.fftpack as fftpack
import scipy.signal as sig


x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Work/sound_corpora/MUMS_converted/DVD 3/STRINGS/CELLOS/CELLO/CelA#2_4.08sec.wav')

T = pydbm.utils.MiscUtils()
H = pydbm.atom.HannGen()

w = 2048
k = 2048
hop = 256

cc = 100

pin = 0
pend = len(x) - w

imp = np.zeros(w * 4)
imp[0] = 1.

out = np.zeros(len(x) * 4)

while pin < pend:
    coefs = T.lpc(x[pin:pin+w], cc)
    i = sig.lfilter([1.], coefs, imp)
    I = fftpack.fft(i, k*4)
    Ir = np.real(I)
    #p = 2*np.pi * np.random.random_sample(w) - np.pi
    p = np.imag(I)
    grain = H.window(w*4) * np.real(fftpack.ifft(Ir + 1j*p, k*4)) 
    out[pin:pin+w*4] += grain
    pin += hop
    

#h = np.random.randn(size) #* sig.lfilter([0.5, 0.7] , 1 ,np.random.randn(size))
#out = H.window(size) * sig.lfilter([1.], coefs, h)

#I = sig.lfilter([1.], coefs, imp)


plt.figure(0)
plt.plot(20*np.log10(abs(I)/len(I))[0:len(I)/2])
plt.show()

#out = sig.fftconvolve(out, h)

out /= max(abs(out))
out *= 0.8

audiolab.wavwrite(out, '/Users/grahamboyes/Desktop/wumpscut.wav', fs)
plt.figure(1)
plt.plot(out)


