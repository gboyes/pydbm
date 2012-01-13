import os

import pydbm.data 
import pydbm.dictionary
import numpy as np
import audiolab


f = open('/Users/grahamboyes/Desktop/sflist', 'rb')
subf = [line.strip('\n') for line in f]
subf = [subf[r] for r in range(0, len(subf), 2)]
list_of_corpora = set([os.path.split(sf)[0] for sf in subf])

x, fs, p = audiolab.wavread('../sounds/harm_fof.wav')
#list_of_corpora = ['/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/HardMallet/piano'] 

S = pydbm.data.SoundDatabase()
for c in list_of_corpora:
    d = pydbm.data.Corpus(c)
    sf = [os.path.split(sf_)[1] for sf_ in subf if os.path.split(sf_)[0] == c]
    #d.getAllSoundfiles()
    d.getSoundfiles(sf)
    #d.getSoundfiles([os.listdir(c)[i] for i in np.random.randint(1, len(os.listdir(c)), 10)]) #get a subset
    S.corpora.append(d)
    
D = pydbm.dictionary.SoundgrainDictionary(fs, S)
n = np.array([0, 1000, 2000, 4000, 8000, 16000, 17000, 18000, 20000, 24000]) 
D.addCorpus(n, 0)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

g = max(abs(x_)) * 0.005

#mod, res, M = D.tvmp(x_.copy(), 100, 35, g)
mod, res, M = D.mp(x_.copy(), 100, 35)
audiolab.wavwrite(mod[0:len(x)], '/Users/grahamboyes/Desktop/norescaleMP.wav', fs)
