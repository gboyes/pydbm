import os

import pydbm.data 
import pydbm.dictionary

import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as audiolab

cf = '/Users/grahamboyes/Desktop/GApresentation/sparse/16k'
C = pydbm.data.InstrumentSoundgrainCorpus(cf)
t = '/Users/grahamboyes/Desktop/GApresentation/targets/R2D2_omsox.mrk2.sdif'

f = os.listdir(cf)

C.getSoundfiles(f, '/Users/grahamboyes/Downloads/Benvibes-sparse.sdif')
S = pydbm.data.SoundDatabase()
S.corpora.append(C)
D = pydbm.dictionary.InstrumentSoundgrainDictionary(16000, S)

p = (D.sdif2array(t, ['XTRD'])['XTRD']['time'] * 16000).astype(int)


D.addCorpus(p, 0)

x, fs, prawn = audiolab.wavread('/Users/grahamboyes/Desktop/GApresentation/targets/R2D2_omsox.wav')


q = np.zeros(max(D.atoms['onset']) + max(D.atoms['duration']))
q[0:len(x)] = x

out, signal, M = D.mpc(q, 100, 0., '/Users/grahamboyes/Downloads/allconstraintsnew.sdif', tv=True, globscalar=0.01)
#out, signal, M = D.mp(q, 41, 0.,)

M.writeSDIF('/Users/grahamboyes/Desktop/GApresentation/models/mod.sdif')

audiolab.wavwrite(out, '/Users/grahamboyes/Desktop/GApresentation/models/mod.wav', 16000) 
