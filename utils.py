import numpy as np
import scipy.linalg as linalg
import scipy.signal as sig
import scipy.fftpack as fftpack

class MiscUtils(object):
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
        '''Return the power spectrum of time domain signal x, NFFT should be evem'''
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

    def mfcc(self, x, Fs, CC, N):
        '''Mel-Frequency cepstral coefficients of signal x'''

        #filterbank parameters
        lowestFrequency = 133.3333
        linearFilters = 13
        linearSpacing = 66.66666666
        logFilters = 27
        logSpacing = 1.0711703
        totalFilters = linearFilters + logFilters

        #frequency band edges
        freqs = lowestFrequency + np.arange(0, linearFilters)*linearSpacing
        freqs = np.r_[(freqs, freqs[linearFilters-1] * logSpacing ** np.arange(1, logFilters+3))]

        lower = freqs[0:totalFilters]
        center = freqs[1:totalFilters+1]
        upper = freqs[2:totalFilters+2]

        mfccWeights = np.zeros((totalFilters, N))
        triangleHeight = 2./(upper-lower)
        farray = np.arange(0., N)/N*Fs

        for chan in range(0, totalFilters):
            a = ((farray > lower[chan])&(farray <= center[chan]))
            c = (farray-lower[chan])/(center[chan]-lower[chan])
            b = ((farray > center[chan]) &(farray <= upper[chan]))
            d = (upper[chan]-farray) / (upper[chan]-center[chan])
            mfccWeights[chan,:] = a * triangleHeight[chan] * c + b * triangleHeight[chan] * d

        w = sig.hamming(len(x))
        coefDCT = 1/np.sqrt(totalFilters/2)*np.cos(((np.arange(0,CC))[:,np.newaxis]) * (2*np.arange(0 ,totalFilters)+1) * np.pi/2/totalFilters)

        #preEmp = sig.lfilter(np.array([1, -.97]), 1, x)

        fr = (np.arange(0, N/2.))[:, np.newaxis]/(N/2.)*Fs/2. 
        j = 0
        for i in range(0, N/2):
            if fr[i] > center[j+1]:
                j += 1
            if j > totalFilters-2:
                j = totalFilters-2
            q = max(1, j + (fr[i] - center[j])/(center[j+1]-center[j]))
            fr[i] = min(totalFilters-.0001, q)
        fri = np.fix(fr).astype(np.int16)
        frac = fr-fri
    
        #fftData = preEmp * w
        fftData = np.array(x)
        fftMag = abs(fftpack.fft(fftData, N))
        earMag = np.log10(np.inner(mfccWeights, fftMag) + 0.0000000001)
        ceps = np.inner(coefDCT, earMag)
        
        return ceps


class TransUtils(object):

    '''Some useful transforms'''

    def stft(self, x, W, NFFT, hop, w='hann'):

        '''Short-Time Fourier Transform                                                                                               
        x -- input signal                                                                                       
        W -- analysis window size                                                                                                 
        NFFT -- FFT size                                                                                                          
        hop -- hop size                                                                                                           
        w -- name of window (hanning, blackman, hamming)                                                                          
        '''

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
           x := signal (anaytical for best results)
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
