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


x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Work/experiments_dup/experiments/test_samples/piano/MPdecomps/threePFNOF_4100_residual.wav')

T = pydbm.utils.MiscUtils()
H = pydbm.atom.HannGen()

size = 16000
coefs = T.lpc(x[16000:17024], 17)
imp = np.zeros(size)
imp[0] = 1.

h = np.random.randn(size) #* sig.lfilter([0.5, 0.7] , 1 ,np.random.randn(size))

#h = np.zeros(size)
#for k in range(1, 40):

#    h += np.cos(2*np.pi*k*220./fs*np.arange(size))

out = H.window(size) * sig.lfilter([1.], coefs, h)

I = sig.lfilter([1.], coefs, imp)

F = fftpack.fft(I)
Fe = np.real(F)

p = 2*np.pi * np.random.random_sample(len(Fe)) - np.pi

out = np.real(fftpack.ifft(Fe + 1j*p))


plt.figure(0)
plt.plot(20*np.log10(abs(fftpack.fft(I))/len(I))[0:len(I)/2])
plt.show()

#out = sig.fftconvolve(out, h)

out /= max(abs(out))
out *= 0.8

audiolab.wavwrite(out, '/Users/grahamboyes/Desktop/weoruy.wav', fs)
plt.figure(1)
plt.plot(out)


#plt.figure(1)
#plt.colorbar()

#plt.xlabel('time (s)')
#plt.ylabel('frequency (Hz)')
