import audiolab

import pydbm_.dictionary
import pydbm_.data
import numpy as np

target = '/Users/grahamboyes/Desktop/AMEN_stereo.wav'
outdir = '/Users/grahamboyes/Desktop'
db = ['/Users/grahamboyes/Desktop/taggedsounds/machine']

xs, fs, p = audiolab.wavread(target)
vecs = (xs[0:, 0], xs[0:, 1])

S = pydbm_.data.SoundDatabase(db)

#if there is a transient model somewhere
n = S.sdif2array('/Users/grahamboyes/Desktop/AMEN_stereo.mrk.sdif', ['XTRD'])['XTRD']
n = (n['time'] * fs).astype(int)

D = pydbm_.dictionary.SoundgrainDictionary(fs, S)

for i in xrange(len(db)):
    D.addCorpus(n, i)

v = []

for ind, x in enumerate(vecs):
    x_ = np.zeros(len(x) + max(D.atoms['duration']))
    x_[0:len(x)] = x
    mod, res, M = D.mp(x_.copy(), 300, 35)
    v.append(mod)

out = np.zeros((len(x), 2))
out[0:, 0] = v[0][0:len(x)]
out[0:, 1] = v[1][0:len(x)]

audiolab.wavwrite(out, '%s/model.wav'%outdir, fs)

