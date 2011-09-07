import os
import numpy as np
import pydbm_.dictionary
import pydbm_.meta
import pydbm_.data
import pydbm_.atom
import pydbm_.utils
import matplotlib.pyplot as plt
import audiolab
import scipy.fftpack as fftpack


x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Sounds/beets/clnkrbrk.wav')

T = pydbm_.utils.MiscUtils()

w = 1024
pin = 0
hop = 256
pend = len(x) - w

num = 33

CC = np.zeros((num, len(x)/float(hop)))

i = 0
while pin < pend:
     CC[0:num, i] = T.mfcc(x[pin:pin+hop], fs, num, 2048)
     i +=1 
     pin += hop


plt.figure(1)
plt.imshow(abs(CC), vmax=1., aspect='auto', origin='lower', cmap='Greys')
plt.colorbar()

plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')