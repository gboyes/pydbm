import time
import copy
import copy

import numpy as np
import scipy.linalg as linalg
import scipy.signal as sig
import scipy.fftpack as fftpack

import pysdif
import audiolab
import numpy.lib.recfunctions as rfn
#import matplotlib.pyplot as plt

import pydbm_.meta
import pydbm_.utils

#idea: deal with statistics of atoms?  Assume a model is a distribution of elements with mean std etc.

class Book(pydbm_.meta.Types, pydbm_.meta.Group, pydbm_.utils.TransUtils):

    '''Time-Frequency synthesis Book'''

    def __init__(self, maxnum, dtype, fs):
        pydbm_.meta.Types.__init__(self)
        self.order = 1
        self.sdifType = 'XBOK'
        self.sampleRate = fs
        self.atoms = np.zeros(maxnum, dtype=dtype)

    def writeSDIF(self, outpath):
        '''Generate an SDIF file from a sequence of decomposition book'''

        f = pysdif.SdifFile('%s'%outpath, 'w')
        f.add_NVT({'date' : time.asctime(time.localtime()), 'sample rate' : self.sampleRate})
        f.add_frame_type('XBOK', 'XGAM NewXGAM, XFOM NewXFOM, XGMM NewXGMM, XDMM NewXDMM')

        self.atoms.sort(order='onset')

        f.add_matrix_type('XGAM', 'index, onset, scale, omega, chirp, mag, phase, norm')
        f.add_matrix_type('XFOM', 'index, onset, scale, omega, chirp, rise, decay, mag, phase, norm')
        f.add_matrix_type('XGMM', 'index, onset, scale, omega, chirp, order, bandwidth, mag, phase, norm')
        f.add_matrix_type('XDMM', 'index, onset, scale, omega, chirp, damp, mag, phase, norm')

        n = 0
        while n < self.num:
            t = self.atoms['onset'][n]
            frame = f.new_frame('XBOK', t /float(self.sampleRate))

            c = 0
            for ind in np.where(self.atoms['onset'] == t)[0]:
                N = self.atoms[ind]

                if N['type'] == 'gabor':
                    frame.add_matrix('XGAM', np.array([[N['index'], N['onset'], N['duration'], N['omega'], N['chirp'], N['mag'], N['phase'], N['norm']]]))

                elif N['type'] == 'FOF':
                    frame.add_matrix('XFOM', np.array([[N['index'], N['onset'], N['duration'], N['omega'], N['chirp'], N['rise'], N['decay'], N['mag'], N['phase'], N['norm']]]))

                elif N['type'] == 'gamma':
                    frame.add_matrix('XGMM', np.array([[N['index'], N['onset'], N['duration'], N['omega'], N['chirp'], N['order'], N['bandwidth'], N['mag'], N['phase'], N['norm']]]))
 
                elif N['type'] == 'damped':
                    frame.add_matrix('XDMM', np.array([[N['index'], N['onset'], N['duration'], N['omega'], N['chirp'], N['damp'], N['mag'], N['phase'], N['norm']]]))

                c += 1

            frame.write()
            n += c

        f.close()

                                                                                                                     
    def shift(self, time_shift, omega_shift):
        '''Shift the contents of a Group in time and frequency                                                                     
           time_shift := num. samples to shift                                                                                                                              
           omega_shift := amount to shift in normalized frequency'''

        C = copy.deepcopy(self)
        C.atoms['onset'] += time_shift
        C.atoms['omega'] += omega_shift

        return C

    def synthesize(self, synthtype='default', kwargs={}):
        out = np.zeros(max(self.atoms['duration']) + max(self.atoms['onset']))
        
        if synthtype == 'default':

            for a in self.atoms:
                a_ = self.atomGenTable[a['type']].gen(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs])
                a_ /= linalg.norm(a_)
                out[a['onset']:a['onset']+a['duration']] += a_ * a['mag']

        elif synthtype == 'FM':

            for a in self.atoms:
                a_ = self.atomGenTable[a['type']].genFM(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs], **kwargs)
                a_ /= linalg.norm(a_)
                out[a['onset']:a['onset']+a['duration']] += a_ * a['mag']
        
        elif synthtype == 'harmonic':
            
            for a in self.atoms:
                a_ = self.atomGenTable[a['type']].gen(a['phase'] + a['k_phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs], **kwargs)
                a_ /= linalg.norm(a_)
                out[a['onset']:a['onset']+a['duration']] += a_ * a['k_mag'] * a['mag']
            
        elif synthtype == 'disintegrate':
            #provide in kwargs a self.num length vector of 'weights'

            q = self.atoms.copy()
            q.sort(order=kwargs['order'])

            for i, a in enumerate(q):
                a_ = self.atomGenTable[a['type']].gen(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs])
                a_ /= linalg.norm(a_)
                out[a['onset']:a['onset']+a['duration']] += a_ * a['mag'] * kwargs['weights'][i]
 
        return out
    
    def wivigram(self, hop, fftsize, plot=False):
        
        W = np.zeros((fftsize, np.ceil(max(self.atoms['onset'] + self.atoms['duration']) / float(hop))))
        
        for a in self.atoms:
            a_ = self.atomGenTable[a['type']].gen(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs])
            a_  /= linalg.norm(a_)
            a_ *= a['mag']
            w = self.dwvd(sig.hilbert(a_), hop, fftsize)
            W[0:fftsize, np.floor(a['onset'] / float(hop)): np.floor(a['onset'] / float(hop)) + np.shape(w)[1]] += np.abs(w)

        if plot:
            p = plt.axes()
            p.imshow(W, interpolation=None, aspect='auto', origin='lower', cmap='Greys', extent=[0., np.shape(W)[1] * float(hop) / self.sampleRate, 0., 0.5 * self.sampleRate])
            p.set_xlabel('Time (sec.)')
            p.set_ylabel("Frequency (Hz.)")

        return W

    def crosscorrelate(self, num, compfunc=lambda x, y: np.abs(np.inner(np.conj(x), y))):
        '''Agglomerate (a subset of) the synthesis book into molecules, where
           num := number of atoms to consider
           thresh := cutoff threshold for inclusion in a molecule 
           outdir := where to write the molecules
           compfunc := a function to compare atoms, default is cross-correlation'''

        S = self.atoms.copy()
        S.sort(order='mag')
        S = S[::-1]
        S = S[0:num]
        
        M = np.zeros((num, num))
        mx = np.max(S['duration'] + S['onset'])

        for i, a in enumerate(S):
            aa = a['mag'] * np.exp(2*np.pi*1j*a['omega']*np.arange(a['duration'])) * np.exp(1j*a['phase']) * self.atomGenTable[a['type']].window(a['duration'])  
            aa /= linalg.norm(aa)
            va = np.zeros(mx, dtype=np.cfloat)
            va[a['onset'] : a['onset']+a['duration']] = aa
 
            for q, b in enumerate(S):
                ab = b['mag'] * np.exp(2*np.pi*1j*b['omega']*np.arange(b['duration'])) * np.exp(1j*b['phase']) * self.atomGenTable[b['type']].window(b['duration'])  
                ab /= linalg.norm(ab)
                vb = np.zeros(mx, dtype=np.cfloat)
                vb[b['onset'] : b['onset'] + b['duration']] = ab
                M[i, q] = compfunc(va, vb)

        return M

    def agglomerate(self, M, thresh):
        '''Agglomerate a cross-correlation matrix'''
        
        #now the Boolean                                                                                
        M_ = M > thresh

        for i, row in enumerate(M_):
            for j, col in enumerate(row):
                if j < i:
                    M_[i, j] = 0
                elif i == j:
                    M_[i, j] = 1

        #get valid molecule indices
        a = [np.array([k for k in xrange(np.shape(M_)[1]) if M_[j][k] == 1]) for j in xrange(np.shape(M_)[0])]
        La = np.array([])
        Na = []
        
        for q in a:
            q = np.setdiff1d(q, La)
            La = np.union1d(La, q)
            if len(q) != 0:
                Na.append(q)

        return Na

    def writeMolecules(self, N, outdir):
        '''Write clustered atoms, where:
           N:= a list of lists of indices that belong in a molecule
           outdir := directory to write to'''
        
        #now build the molecules and the reconstruction
        allout = np.zeros(len(self.model))
        for i, mol in enumerate(Na):
            out = np.zeros(max(self.atoms['onset'][mol] + self.atoms['duration'][mol]))
    
            for ind in mol:
                a_ = self.atomGenTable[self.atoms[ind]['type']].gen(self.atoms[ind]['phase'], *[self.atoms[ind][arg] for arg in self.atomGenTable[self.atoms[ind]['type']].genargs])
                a_ /= linalg.norm(a_)
                out[self.atoms['onset'][ind]:self.atoms['onset'][ind]+self.atoms['duration'][ind]] += a_ * self.atoms[ind]['mag']
                
            audiolab.wavwrite(out, '%s/molecule%i.wav'%(outdir, i), self.sampleRate)
            allout[0:len(out)] += out 

        audiolab.wavwrite(allout, '%s/all_molecules.wav'%outdir, self.sampleRate)
        

#a book of quasi-harmonically related atoms (components of structures)

class SpectralBook(pydbm_.utils.MiscUtils, Book):
    '''A synthesis book with a description of spectral structures'''

    def __init__(self, maxnum, dtype, fs):
        pydbm_.meta.Types.__init__(self)
        self.order = 2
        self.sdifType = 'XHBK'
        self.sampleRate = fs
        self.atoms = np.zeros(maxnum, dtype=dtype)

    def synthesize(self, synthtype='default', kwargs={}):
        out = np.zeros(max(self.atoms['duration']) + max(self.atoms['onset']))
        
        if synthtype == 'default':

            for a in self.atoms:
                a_ = self.atomGenTable[a['type']].gen(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs])
                a_ /= linalg.norm(a_)
                out[a['onset']:a['onset']+a['duration']] += a_ * a['mag']

        elif synthtype == 'FM':

            s = np.unique(self.atoms['index'][1])
            fdepth = kwargs['depth']

            for k in s:
                inds = np.where(self.atoms['index'][1] == k)[0]
                f0 = self.atoms['omega'][inds[0]]

                for i in inds:
                    a = self.atoms[i]
                    kwargs['depth'] = (a['omega'] / f0) * fdepth 
                    a_ = self.atomGenTable[a['type']].genFM(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs], **kwargs)
                    a_ /= linalg.norm(a_)
                    out[a['onset']:a['onset']+a['duration']] += a_ * a['mag']

        return out

    def structSynth(self, k):

        '''Synthesize a given structure
           k := index of structure to synthesize'''

        ind = np.where(self.atoms['index'][1] == k)[0]
        out = np.zeros(self.atoms['duration'][ind[0]])
        
        for a in self.atoms[ind]:
            a_ = self.atomGenTable[a['type']].gen(a['phase'], *[a[arg] for arg in self.atomGenTable[a['type']].genargs])
            a_ /= linalg.norm(a_)
            out += a_ * a['mag']

        return out

    def selfSimilarity(self):
        '''Compute spectral self-similarity matrix using Itakura-Saito distance'''

        K = np.unique(self.atoms['index'][1])
        I = np.zeros((len(K), len(K)))

        for i in K: 
            a = self.structSynth(i)
            for j in K: 
                b = self.structSynth(j)
                m = max([len(a), len(b)])
                I[i, j] = self.itakuraSaito(self.powerSpectrum(a, m), self.powerSpectrum(b, m))

        return I 

    def writeMolecules(self, N, outdir):
        '''Write clustered atoms, where:
           N:= a list of lists of indices that belong in a molecule
           outdir := directory to write to'''
        
        #now build the molecules and the reconstruction
        allout = np.zeros(len(self.model))

        for i, mol in enumerate(N):
            out = np.zeros(len(self.model))
    
            for ind in mol:
                q = np.where(self.atoms['index'][1]==ind)[0]

                out[self.atoms['onset'][q[0]]:self.atoms['onset'][q[0]]+self.atoms['duration'][q[0]]] += self.structSynth(ind)
                
            audiolab.wavwrite(out, '%s/molecule%i.wav'%(outdir, i), self.sampleRate)
            allout[0:len(out)] += out 

        audiolab.wavwrite(allout, '%s/all_molecules.wav'%outdir, self.sampleRate)

    #visualization
    def pianoroll(self, dsfactor):

        '''A piano roll-like visualization of a SpectralBook's contents'''

        out = np.zeros((len(self.model) / dsfactor, max(self.hz2midi(self.atoms['omega'] * self.sampleRate))))
        s = np.unique(self.atoms['index'][1])

        for k in s:
            inds = np.where(self.atoms['index'][1] == k)[0]
            f0 = np.round(self.hz2midi(self.atoms['omega'][inds[0]] * self.sampleRate))
            out[np.floor((self.atoms['onset'][inds[0]])/dsfactor): np.floor((self.atoms['onset'][inds[0]]) / dsfactor) + np.floor(self.atoms['duration'][inds[0]] / dsfactor), f0] += np.ones(np.floor(self.atoms['duration'][inds[0]] / dsfactor)) * sum(self.atoms['mag'][inds])

        return out.T

            
