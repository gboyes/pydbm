import numpy as np
import pydbm.utils 
import matplotlib.pyplot as plt
import time

u = pydbm.utils.Utils()
x = np.random.randn(100000)

hop = 256
nwin = 1024
nfft = 2048

num = 1000

c = 0

a = time.clock() 
while c < num:
    u.stft(x, nwin, nfft, hop)
    c+=1
print(time.clock() - a)

c = 0
a = time.clock() 
while c < num:
    u.stft_(x, nwin, nfft, hop)
    c+=1
print(time.clock() - a)
