import os
import numpy as np
import pydbm.dictionary
import pydbm.meta
import pydbm.data
import pydbm.atom
import pydbm.utils
import matplotlib.pyplot as plt
import scikits.audiolab
import scipy.fftpack as fftpack

import time

import scikits.talkbox.features as stf

x, fs, p = scikits.audiolab.wavread('/Users/grahamboyes/Documents/Sounds/beets/clnkrbrk.wav')

#x = np.random.randn(len(x))

T = pydbm.utils.MiscUtils()

w = 1024
pin = 0
hop = 160
pend = len(x) - w

num = 10

a = time.clock()
CC = T.mfcc(x, fs, w, hop, 2048, num)[0]
print(time.clock()-a)

'''
a = time.clock()
CC = np.zeros((num, len(x)/float(hop)))

i = 0
while pin < pend:
     CC[0:num, i] = T.mfcc_(x[pin:pin+w], fs, num, 2048)
     i +=1 
     pin += hop
print(time.clock()-a)
'''

b = time.clock()
CC2 = stf.mfcc(x, nwin=w, nfft=2048, fs=fs, nceps=num)[0]
print(time.clock() - b)

plt.figure(1)
plt.imshow(abs(CC.T), vmax=1., aspect='auto', origin='lower')
plt.colorbar()

plt.figure(2)
plt.imshow(abs(CC2.T), vmax=1., aspect='auto', origin='lower')
plt.colorbar()

plt.xlabel('time (s)')
plt.ylabel('mfcc')
