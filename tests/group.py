import pydbm_.dictionary
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import audiolab

x, fs, p = audiolab.wavread('/Users/grahamboyes/Desktop/VibesMedium_7100-80.wav')

D = pydbm_.dictionary.Dictionary(fs)
S = pydbm_.dictionary.SpecDictionary(fs)

dtype = 'hann'
dur = 4096
hop = 128
tmin = 0
tmax = len(x)-dur 
winargs = {}

D.addSTFT(dtype, dur, 128, tmin, tmax, **winargs)
D.index()

#dtype, midicents, scales, onsets, **kwargs)
#S.addSTFT(dtype, dur, 128, tmin, tmax, **winargs)
S.addNote(dtype, 7100, [dur, 1024], [np.arange(0, tmax, hop), np.arange(0, len(x)-1024, hop/2)], **winargs)
S.index()

#signal, cmax, srr_thresh, tolmidicents, maxPeaks, dBthresh, overspec

model, res, sBook = S.mp(x.copy(), 1000, 35., 10000, 5, -80.0, 1)
audiolab.wavwrite(model, '/Users/grahamboyes/Desktop/VibesMedium_7100-80_mod.wav', fs)