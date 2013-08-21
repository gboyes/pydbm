import numpy as np
import scipy.fftpack as fftpack
import pydbm.utils
import time

def fftComp(N, num):
    U = pydbm.utils.Utils()
    Nfft = 2**(U.nextpow2(N))
    x = np.random.randn(N)
    y = np.random.randn(N)

    t = time.time()
    X = fftpack.fft(x, Nfft)

    for n in range(num):  
        np.real(fftpack.ifft(np.conj(fftpack.fft(y, Nfft)) * X))

    q = time.time()
    print(q-t)

def tdComp(N, num):
    x = np.random.randn(N)
    y = np.random.randn(N)
    q = np.zeros(N*2)
    q[0:N] = x
    t = time.time()
    for n in range(num):
        for i in range(N):
            np.inner(y, q[i:i+N])
    p = time.time()
    print(p-t)

if __name__ == '__main__':
     K = 16385
     r = 16384
     fftComp(K, r)
     tdComp(K, r/256)
     
        

