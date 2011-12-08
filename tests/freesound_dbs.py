import audiolab
import pydbm.dictionary
import pydbm.data
import numpy as np


target = '/Users/grahamboyes/Desktop/AMENS/AMEN.wav'
db = ['']
outdir = '/Users/grahamboyes/Desktop'

#a decomposition using the outdir                                                                                                                 
x, fs, p = audiolab.wavread(target)

S = pydbm.data.SoundDatabase(db)

#if there is a transient model somewhere
n = S.sdif2array('/Users/grahamboyes/Desktop/AMEN.mrk.sdif', ['XTRD'])['XTRD']
n = (n['time'] * fs).astype(int)

D = pydbm.dictionary.SoundgrainDictionary(fs, S)

for i in xrange(len(db)):
    D.addCorpus(n, i)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

mod, res, M = D.mp(x_.copy(), 300, 35)
audiolab.wavwrite(mod[0:len(x)], '%s/model.wav'%outdir, fs)

