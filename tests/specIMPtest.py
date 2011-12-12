import numpy as np
import pydbm.dictionary
import matplotlib.pyplot as plt
import scikits.audiolab
import time

x, fs, p = scikits.audiolab.wavread('/home/frobenius/Documents/Sounds/harm_fof.wav')

D = pydbm.dictionary.SpecDictionary(fs)
D.addNote('hann', 6000, [512], [np.arange(0, len(x)-512, 256)])
a = time.clock()
#mod, res, book = D.mp(x, 1000, 35, 10000, 10, -80, 2) 
mod, res, book = D.mp2(x, 1000, 35, 10000, 10, -80, 2) 
print(time.clock()-a)
print(book.num())

scikits.audiolab.wavwrite(mod, '/home/frobenius/Desktop/mod.wav', fs)
