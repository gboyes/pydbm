import os
import numpy as np
import pydbm_.dictionary
import pydbm_.meta
import pydbm_.data
import pydbm_.atom
import pydbm_.utils

#import matplotlib.pyplot as plt
import audiolab

import pysdif
import scipy.fftpack as fftpack
import scipy.signal as sig

sdif_in = '/Users/grahamboyes/Documents/Work/project_m/harm_fof_shape_poly.sdif'
x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Work/project_m/harm_fof.wav')

S = pydbm_.data.PolygonGroup(sdif_in)

s1 = S.polygons[1]

p = s1.getPolyHull(16000, 128, 2048)

#plt.scatter(s1.polyHull['hop'], s1.polyHull['bin'])
#plt.hold(True)
#plt.scatter(s1.tfPoints['hop'], s1.tfPoints['bin'], c='r')

D = pydbm_.dictionary.Dictionary(fs)

ps = [(1024, 2048, 128), (256, 512, 64), (512, 4096, 128)]
kwargs = {}

for ls in ps:

    D.addPolygon(s1, 'hann', *ls, **kwargs)

mod, res, book = D.mp(x.copy(), 100, 35)
audiolab.wavwrite(mod, '/Users/grahamboyes/Desktop/foo.wav', fs)