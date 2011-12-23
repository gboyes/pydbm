import numpy as np
import pydbm.dictionary
import scikits.audiolab as audiolab

x, fs, p = audiolab.wavread('/home/frobenius/Documents/Sounds/harm_fof.wav')
list_of_corpora = ['/home/frobenius/Documents/Sounds/16khz/ordinario/HardMallet/forte']

S = pydbm.data.SoundDatabase(list_of_corpora)
D = pydbm.dictionary.SoundgrainDictionary(fs, S)

#'base' quantization
onsets = np.arange(0, len(x), 500)

for k in range(len(list_of_corpora)):
    D.addCorpus(onsets, k)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

#min spacing is a quantizing element
mod, res, M = D.mpc(x_.copy(), 1000, 35, 4, 500)
audiolab.wavwrite(mod, '/home/frobenius/Desktop/mod.wav', fs)
