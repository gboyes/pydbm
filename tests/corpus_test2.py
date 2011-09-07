import pydbm_.data 
import pydbm_.dictionary

import numpy as np
import audiolab

x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Work/project_m/harm_fof.wav')
sdif = '/Users/grahamboyes/Documents/Work/project_m/harm_fof_shape.sdif'

list_of_corpora = ['/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/forte', '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/HardMallet/forte']

Y = pydbm_.data.PolygonGroup(sdif)
S = pydbm_.data.SoundDatabase(list_of_corpora)

D = pydbm_.dictionary.SoundgrainDictionary(fs, S)

for y in Y.polygons:
    D.addPolygon(x.copy(), 1024, y, 2048, 256, 6)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

mod, res, M = D.mp2(x_.copy(), 30, 35)
audiolab.wavwrite(mod, '/Users/grahamboyes/Desktop/prawn.wav', fs)