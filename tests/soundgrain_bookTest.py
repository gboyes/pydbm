import pydbm_.data 
import pydbm_.dictionary
import pydbm_.book

import audiolab
import numpy as np

x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/Work/project_m/harm_fof.wav')

list_of_corpora = ['/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/forte', '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/piano']
S = pydbm_.data.SoundDatabase(list_of_corpora)

D = pydbm_.dictionary.SoundgrainDictionary(16000, S)
D.addCorpus(np.array([512, 1024, 2048, 4096]), 0)
D.addCorpus(np.array([512, 1024, 2048, 4096]), 1)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

mod, res, B = D.mp(x_.copy(), 30, 10)
audiolab.wavwrite(mod, '/Users/grahamboyes/Desktop/prawnprollip.wav', fs)

sout = '/Users/grahamboyes/Desktop/testDIF.sdif'
B.writeSDIF(sout)

out = B.synthesize()
audiolab.wavwrite(out, '/Users/grahamboyes/Desktop/prawnprollipsynth.wav', fs)