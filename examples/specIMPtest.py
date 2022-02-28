import numpy as np
import pydbm.dictionary
from scipy.io import wavfile
import time

fs, x = wavfile.read('./sounds/harm_fof.wav')
x = x / 32768.0

D = pydbm.dictionary.SpecDictionary(fs)
D.addNote(b'hann', 6000, [512], [np.arange(0, len(x)-512, 256)])
a = time.time()
mod, res, book = D.mp2(x, 1000, 35, 10000, 10, -80, 2)
print('Decomposition took {}'.format(time.time()-a))
print('Synthesis book contains {} elements'.format(book.num()))

wavfile.write('./sounds/spec-test-model.wav', fs, mod)
wavfile.write('./sounds/spec-test-residual.wav', fs, res)
