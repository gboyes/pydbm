'''pydbm : a python library for dictionary-based methods 
    Copyright (C) 2011 Graham Boyes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import scipy.linalg as linalg
import scipy.signal as sig
import scipy.fftpack as fftpack

class Utils(object):
    '''Class for related utilities'''
    
    def zeroNorm(self, x):
        '''Compute the L-zero norm of x, i.e. count its non-zero elements'''
        return np.sum(x != 0.)

    def ed(self, x):
        '''Return the Energy Density of signal x(n), after Gabor 1947 (mostly for instructive purposes)'''
        psi = sig.hilbert(x)    #alternatively, inverse transform of X(f) with negative frequencies set to zero
        psipsi = psi*np.conj(psi)
        return psipsi

    def sped(self, x):
        '''Return the Spectral Energy Density of signal x(n), after Gabor 1947 (mostly for instructive purposes)'''
        psi = sig.hilbert(x)
        phi = fftpack.fft(psi)
        phiphi = phi*np.conj(phi)
        return phiphi

    def aLog2(self, x):
        '''Returns f,s such that x = f * 2**s for x as a single element'''
        x = np.real(x)
        q = (x==0)
        f = abs(x) + q
        z = (x!=0)
        s = (np.floor(np.log(f)/np.log(2))+1)*z
        f = np.sign(x) * f / (2**s)
        return f, s

    def nextpow2(self, n):
        '''Return the next power of two larger than a number n'''
        p, q = self.aLog2(abs(n))
        if p == 0.5:
            q = q-1
        return int(q)

    #measurements
    def shannonEntropy(self, alpha):
        '''Return the Shannon entropy of vector alpha[n]'''  
        return -1 * (np.sum(np.abs(alpha)**2 * np.log2(np.abs(alpha)**2)))

    def renyiEntropy(self, alpha, beta):
        '''Return the Renyi entropy of vector alpha[n], with coefficient beta'''
        return 1./(beta-1) * np.log2(np.sum(np.abs(alpha)**(2*beta)))
        

    #conversion
    def hz2midi(self, cps):
        '''Linear frequency to MIDI note conversion
        cps -- Frequency in Hz.'''

        m = 69 + 12*np.log2(cps/440. + 0.0000000001)

        return m

    def midi2hz(self, midi_num, f_ref=440.0, midi_ref=69):
        '''MIDI-to-frequency converter'''
    
        return f_ref * 2**( (midi_num-midi_ref) / 12.)

    def crest(self, x):
        '''Crest factor of ndarray x'''
        return np.max(np.abs(x)) / np.sqrt(np.sum(x**2)/float(len(x)))

    def adaptiveMarkers(self, x, hop, min_spacing, max_spacing, mode='diff'):
        '''Adaptively set determine time-domain markers for a signal, where
           x := signal
           hop := analysis hop size
           min_spacing := minimum spacing of markers
           max_spacing := maximum spacing of markers
           mode := amplitude profile used to generate markers'''

        if mode == 'diff':
            x_hat = np.diff(np.real(self.ed(x)))

        elif mode == 'density':
            x_hat = np.real(self.ed(x))

        elif mode == 'signal':
            x_hat = abs(x)

        else:
            raise Exception("mode does not exist")

        numFrames = (len(x_hat) / hop) + 1 
        frames = np.zeros(numFrames)
        zd = np.zeros(numFrames * hop)
        zd[0:len(x_hat)] += x_hat

        n = 0
        i = 0
        while n < len(zd):
            frames[i] = np.sqrt(np.sum(zd[n:n+hop]**2)/hop)
            i+=1
            n+=hop

        mi = min(frames)
        ma = max(frames)

        k = []

        for indx, val in enumerate(frames):
            inv = np.round(np.interp(val, [mi, ma], [max_spacing, min_spacing]))
            r = np.arange(indx * hop, indx * hop + hop, inv)
            k = np.union1d(k, r)

        return  k.astype(int)

    def powerSpectrum(self, x, NFFT):
        '''Return the power spectrum of time domain signal x, NFFT should be even'''
        #return (abs(fftpack.fft(x, NFFT)) / (NFFT/2.))[0:NFFT/2]**2
        return (abs(fftpack.fft(x, NFFT)) / (NFFT))[0:NFFT]**2
        

    def itakuraSaito(self, X, Y):
        '''Itakura - Saito distance, where
           X := a K-length power spectrum
           Y := a K-length power spectrum
           i.e. (abs(fftpack.fft(x, len(x))) / (len(x)/2.))[0:len(x)/2]'''

        return (1./len(X)) * sum( ( X / Y) - np.log( X / Y ) - 1)

    def levinson(self, acf, order):
        '''Levinson recursion for a given autocorrelation function and order'''
        
        R = linalg.toeplitz([acf[0:order]], [(acf[0:order]).conjugate()])
        b = -acf[1:order+1]
        a = linalg.lstsq(R, b.T)[0].T
        
        return np.real(np.r_[(1., a)])

    def lpc(self, x, order):
        '''Generate a given number of linear prediction coefficients for a given signal'''
        
        #compute biased autocorrelation function
        S = fftpack.fft(x,(2 ** self.nextpow2(2.*len(x))))
        acf = fftpack.ifft(abs(S**2))
        acf = acf/len(x)

        if linalg.norm(acf) != 0:
            return  self.levinson(acf, order)
        else:
            return  np.r_[(1., np.zeros(order))]


    def trfbank(self, fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
        """Compute triangular filterbank for MFCC computation."""
        
        # Total number of filters
        nfilt = nlinfilt + nlogfilt
        freqs = np.zeros(nfilt+2)
        freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
        freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
        heights = 2./(freqs[2:] - freqs[0:-2])

        # Compute filterbank coeff (in fft domain, in bins)
        fbank = np.zeros((nfilt, nfft))
        # FFT bins (in Hz)
        nfreqs = np.arange(nfft) / (1. * nfft) * fs

        for i in range(nfilt):
            low = freqs[i]
            cen = freqs[i+1]
            hi = freqs[i+2]

            lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
            lslope = heights[i] / (cen - low)
            rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
            rslope = heights[i] / (hi - cen)
            fbank[i][lid] = lslope * (nfreqs[lid] - low)
            fbank[i][rid] = rslope * (hi - nfreqs[rid])

        return fbank, freqs                                      

    def mfcc(self, x, Fs, win, hop, nfft, CC):
        '''Mel-Frequency cepstral coefficients of a signal
           x := signal
           Fs := sampling rate
           win := window size
           hop := hop size
           nfft := fft size
           CC := number of coefficients'''

        #filterbank parameters
        lowestFrequency = 133.3333
        linearFilters = 13
        linearSpacing = 200./3
        logFilters = 27
        logSpacing = 1.0711703
        totalFilters = linearFilters + logFilters

        w = sig.hamming(win)

        seg = np.zeros((np.ceil(len(x)/float(hop)), win))
        i = 0
        pin = 0
        pend = len(x) - win
        while pin < pend:
            seg[i, 0:win] = x[pin:pin+win] * w
            i += 1
            pin += hop
     
        preEmp = sig.lfilter(np.array([1, -.97]), 1, seg)
        fbank = self.trfbank(Fs, nfft, lowestFrequency, linearSpacing, logSpacing, linearFilters, logFilters)[0]
        fftMag = np.abs(fftpack.fft(preEmp, nfft, axis=-1))
        earMag = np.log10(np.inner(fftMag, fbank) + 0.0000000001)
        ceps = fftpack.realtransforms.dct(earMag, type=2, norm='ortho', axis=-1)[:, 0:CC]
        
        return ceps, earMag

    def segmat(self, x, nwin, hop):
        '''Make a segment matrix out of a vector
           x := vector
           nwin := window size
           hop := hop size'''

        X = np.zeros((nwin, np.ceil(len(x)/float(hop))))
        pin = 0
        pend = len(x) - nwin
        while pin < pend:
            X[:, pin/hop] = x[pin:pin+nwin]
            pin += hop    
        
        return X

    def stft(self, x, nwin, nfft, hop, w='hann'):
        '''Short-time Fourier transform
           x := signal
           nwin := window size
           nfft := fft size
           hop := hop size
           w := window type, one of ['hann', 'blackman', 'hamming']'''  

        windows = {'hann' :sig.hanning, 'blackman' : sig.blackman, 'hamming' : sig.hamming}
        win = windows[w](nwin)
        xs = self.segmat(x, nwin, hop) * np.vstack(win)  
        X = fftpack.fft(xs, nfft, axis=0)
        return X

    def stft_(self, x, W, NFFT, hop, w='hann'):
        #this one is actually slightly faster
        '''Short-time Fourier transform
           x := signal
           nwin := window size
           nfft := fft size
           hop := hop size
           w := window type, one of ['hann', 'blackman', 'hamming']'''  

        L = len(x)
        x = np.concatenate((np.zeros(W), x, W - np.zeros(np.mod(L, hop))))
        num = len(x)/hop
        X = np.zeros((NFFT, num), dtype=np.cfloat)

        windows = {'hann' :sig.hanning, 'blackman' : sig.blackman, 'hamming' : sig.hamming}
        w = windows[w](W)

        pin = 0
        pend = len(x)-W
        cnt = 0

        while pin < pend:
            g = x[pin : pin+W] * w
            G = fftpack.fft(g, NFFT)
            X[0:NFFT, cnt] += G
            cnt += 1
            pin += hop
        X = X[0:NFFT, np.floor(W/float(hop))/2 : np.floor(L/float(hop))]
        return X
        
    def dftMatrix(self, N):
        '''N point Discrete Fourier Transform Matrix'''
        f = np.arange(N)

        #Vandermonde matrix (N, N)                                                                                                                          
        F = np.outer(f, f)

        #basis                                                                                                                                               
        omega = np.exp(-2*np.pi*1j/N)
        W = np.asmatrix(1./np.sqrt(N)*np.power(omega, F))

        return W

    def dct(self, x):
        '''Compute discrete cosine transform of 1-d array x'''

        #probably don't need this here anymore since it is in fftpack now    

        N = len(x)
        #calculate DCT weights
        w = (np.exp(-1j*np.arange(N)*np.pi/(2*N))/np.sqrt(2*N))
        w[0] = w[0]/np.sqrt(2)

        #test for odd or even function 
        if (N%2==1) or (any(np.isreal(x)) == False):
            y = np.zeros(2*N)
            y[0:N] = x
            y[N:2*N] = x[::-1]
            yy = fftpack.fft(y)
            yy = yy[0:N]
        else:
            y = np.r_[(x[0:N:2], x[N:0:-2])]
            yy = fftpack.fft(y)
            w = 2*w

        #apply weights
        X = w * yy

        if all(np.isreal(x)):
            X = np.real(X)

        return X

        
    def idct(self, X):
        '''Compute inverse discrete cosine transform of a 1-d array X'''
        
        N = len(X)
        w = np.sqrt(2*N)*np.exp(1j*np.arange(N)*np.pi/(2.*N))

        if (N%2==1) or (any(np.isreal(X))== False):
            w[0] = w[0] * np.sqrt(2)
            yy = np.zeros(2*N)
            yy[0:N] = w * X
            yy[N+1:2*N] = -1j * w[1:N] * X[1:N][::-1]
            y = fftpack.ifft(yy)
            x = y[0:N]
        else:
            w[0] = w[0]/np.sqrt(2)
            yy = X *w
            y = fftpack.ifft(yy)
            x = np.zeros(N)
            x[0:N:2] = y[0:N/2]
            x[1:N:2] = y[N:(N/2)-1:-1]

        if all(np.isreal(x)):
            x = np.real(x)

        return x

    def dwvd(self, x, hop, fftsize):
        '''Discrete Wigner-Ville distribution
           x := signal (analytic for best results)
           hop := time-domain sampling factor
           fftsize := the number of spectral samples (note that the frequency of the kth bin is (0.5 * (k/fftsize) *  fs)''' 

        t = np.arange(0, len(x), hop)
        W = np.zeros((fftsize, len(t)), dtype=np.cfloat)

        for i, ti in enumerate(t):
            taumax = min([ti-1, len(x) - ti - 1, np.round(fftsize/2)-1])
            tau = np.arange(-taumax, taumax)
            indices = np.remainder(fftsize+tau, fftsize)
            W[indices, i] = x[ti+tau-1] * np.conj(x[ti-tau-1])
            tau = np.round(fftsize/2)
    
            if (ti <= len(x) - tau) and (ti >= tau):
                W[tau, i] = 0.5 * (x[ti+tau - 1] * np.conj(x[ti-tau-1])  + x[ti-tau-1] * np.conj(x[ti+tau-1]))

        return fftpack.fft(W, axis=0)
