import pydbm.data 
import pydbm.dictionary
import pydbm.meta

import numpy as np
import scikits.audiolab as audiolab

x, fs, p = audiolab.wavread('/home/geb/Documents/projects/Spielraum/targets/Syncytium/Syncytium-16kHz.wav')
sdif = '/home/geb/Documents/projects/Spielraum/targets/Syncytium/Syncytium-mrk.sdif'

#list_of_corpora = ['/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/pizzicato-bartok', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/pizzicato-l-vib', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/artificial-harmonic', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Violin/instrumental/artificial-harmonic', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Violin/instrumental/pizzicato-bartok', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Violin/instrumental/pizzicato-l-vib']

#list_of_corpora = ['/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/artificial-harmonic-tremolo'] 
list_of_corpora = ['/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Violin/instrumental/artificial-harmonic-tremolo', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/artificial-harmonic-tremolo', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Cello/instrumental/artificial-harmonic', '/home/geb/Documents/projects/Spielraum/Dictionaries/16kHz/Violin/instrumental/artificial-harmonic']

#cpb, cpv, cps, cah
#vpb, vpv, vps, vah

I = pydbm.meta.IO()
S = pydbm.data.SoundDatabase(list_of_corpora)
D = pydbm.dictionary.SoundgrainDictionary(fs, S)

#'base' quantization                                                                                                                                         
onsets = I.sdif2array(sdif, ['XTRD'])['XTRD']['time'] * fs
onsets = onsets.astype(int) 

for k in range(len(list_of_corpora)):
    D.addCorpus(onsets, k)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x


#for y in Y.polygons:
#    D.addPolygon(x.copy(), 1024, y, 2048, 256, 6)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

maxiter = 1000
maxsimul = 6
mindist = 7160

#mod, res, M = D.mp(x_.copy(), 10, 35)
mod, res, M = D.mpc(x_.copy(), maxiter, 35, maxsimul, mindist)
mod /= max(abs(mod))
mod *= 0.90
audiolab.wavwrite(mod, '/home/geb/Desktop/Spiel_%i_%i_%i.wav'%(maxiter, maxsimul, mindist), fs)
M.writeSDIF('/home/geb/Desktop/Spiel_%i_%i_%i.sdif'%(maxiter, maxsimul, mindist))
