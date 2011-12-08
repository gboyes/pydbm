import pydbm.data 
import pydbm.dictionary

import numpy as np
import audiolab

x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/python/pythonDafx/SOUNDS/harm_fof.wav')
sdif = '/Users/grahamboyes/Documents/Work/project_m/harm_fof_shape.sdif'

list_of_corpora = ['/Users/grahamboyes/Documents/Work/sound_corpora/16k/ModernGrand/AF/ordinario_velshort/chromatic']

#Y = pydbm.data.PolygonGroup(sdif)
S = pydbm.data.SoundDatabase(list_of_corpora)

D = pydbm.dictionary.SoundgrainDictionary(fs, S)

#for y in Y.polygons:
#    D.addPolygon(x.copy(), 1024, y, 2048, 256, 6)

n = S.sdif2array('/Users/grahamboyes/Desktop/ballard.mrk.sdif', ['XTRD'])['XTRD']
n = (n['time'] * fs).astype(int)


n = np.array([0, 1000, 2000, 4000, 8000, 16000, 17000, 18000, 20000, 24000]) 
D.addCorpus(n, 0)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

#global scalar
g = max(abs(x_)) * 0.05

mod, res, M = D.mp2(x_.copy(), 100, 35, g)
#mod, res, M = D.mp(x_.copy(), 100, 35)
audiolab.wavwrite(mod[0:len(x)], '/Users/grahamboyes/Desktop/norescaleMP.wav', fs)
