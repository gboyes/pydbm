import os
import numpy as np
import pydbm_.dictionary
import pydbm_.meta
import pydbm_.data
import pydbm_.atom
import pydbm_.utils

import matplotlib.pyplot as plt
import audiolab

import pysdif
import scipy.fftpack as fftpack
import scipy.signal as sig

sdif_in = '/Users/grahamboyes/Documents/Work/project_m/some-regions2.sdif'

S = pydbm_.data.PolygonGroup(sdif_in)
s1 = S.polygons[0]
p = s1.getPolyHull(16000, 512, 128)

plt.scatter(s1.polyHull['hop'], s1.polyHull['bin'])
plt.hold(True)
plt.scatter(s1.tfPoints['hop'], s1.tfPoints['bin'], c='r')