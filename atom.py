'''pydbm : a python library for dictionary-based methods 
    Copyright (C) 2011-2015 Graham Boyes

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
import scipy.fftpack as fftpack
import atom_

#Atom generators#
#################

class Window(object):
    '''Abstract base class for window-based atoms'''

    def enbw(self, N, **winargs):
        '''Equivalent noise bandwidth (in bins) for a given window, where:
           N := size of the window
           winargs := additional window arguments'''

        w = self.window(N, **winargs)

        return N * np.sum(w**2) / np.sum(w)**2

    def coherentGain(self, N, **winargs):
        '''Coherent gain value associated with the window in question, where:
           N := size of the window
           winargs := additional window arguments'''

        return abs(fftpack.fft(self.window(N, **winargs))[0]) / N

    def minHop(self, N, **winargs):
        '''An estimate of the minimum hop expressed as a fraction of the window size, where:
           N := size of the window
           winargs := additional window arguments'''

        return N / (2 * N * self.enbw(N, **winargs))

class Sinusoid(object):
    '''Base class for sinusoid-type atoms'''

    def realSinusoid(self, N, omega, chirp, phi):
        return np.cos(2 * np.pi * (omega + 0.5*chirp*np.arange(N)) * np.arange(N) + phi)

    def complexSinusoid(self, N, omega):
        return np.exp(2 * np.pi * omega * np.arange(N))

    def vibratingSinusoid(self, N, omega, phi, omega_m, phi_m, depth):
        return np.cos(2 * np.pi * omega * np.arange(N) + (depth/omega_m * np.sin(2 * np.pi * omega_m * np.arange(N) + phi_m)) + phi)

setattr(Sinusoid, 'realSinusoid_', atom_.realSinusoid_)

class HarmonicSinusoid(Sinusoid):

    def realHarmonicSinusoid_parametric(self, N, omega, phi, numPartials, omega_function=lambda f, k: k*f, amp_function=lambda k: np.exp(k), phi_function=lambda phi: phi):

        out = np.zeros(N)
        for i in xrange(1, numPartials+1):
            out += self.realSinusoid_(N, omega_function(omega, i), phi_function(phi)) * amp_function(i)
        return out

    def realHarmonicSinusoid(self, N, amp_list, omega_list, phi_list, maxnum=1000):
        return sum([amp_list[i] * self.realSinusoid_(N, omega_list[i], phi_list[i]) for i in xrange(min((len(omega_list), maxnum)))])   

setattr(HarmonicSinusoid, 'realSinusoid_', atom_.realSinusoid_)


class HannGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'hann'
        self.dictionarySdifType = 'XHAN'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float)]
        self.bookSdifType = 'XHAM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp']
        self.winargs = []

    def window(self, N):
        return 0.5 * (1 - np.cos( 2*np.pi*np.arange(N) / (N-1)))

setattr(HannGen, 'gen', atom_.hann_)
setattr(HannGen, 'genFM', atom_.hannFM_)

class BlackmanGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'blackman'
        self.dictionarySdifType = 'XBLK'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float)]
        self.bookSdifType = 'XBLM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp']
        self.winargs = []

    def window(self, N):
        return 0.42 - 0.5 * np.cos(2*np.pi*np.arange(N)/(N-1)) + 0.08 * np.cos(4*np.pi*np.arange(N)/(N-1))

setattr(BlackmanGen, 'gen', atom_.blackman_)
setattr(BlackmanGen, 'genFM', atom_.blackmanFM_)

class GaborGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'gabor'
        self.dictionarySdifType = 'XGAB'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float)]
        self.bookSdifType = 'XGAM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp']
        self.winargs = []

    def window(self, N, alpha=0.125):
        '''Gaussian distribution
            N := length of vector
            alpha := variance coef.'''
        
        return np.exp(-( ( np.arange(N) - N/2. )**2) / (2. * ( (alpha * N)**2)))

#the compiled generator    
setattr(GaborGen, 'gen', atom_.gabor_)
setattr(GaborGen, 'genFM', atom_.gaborFM_)

class GammaGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'gamma' 
        self.dictionarySdifType = 'XGMA'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('bandwidth', float), ('order', float)]
        self.bookSdifType = 'XGMM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('bandwidth', float), ('order', float), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('bandwidth', float), ('order', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp', 'order', 'bandwidth']
        self.winargs = ['order', 'bandwidth']

    def window(self, N, order=20., bandwidth=0.01):
        '''Gamma distribution
            N := size
            order := order of function
            bandwidth := spectral width of distribution (norm. freq.)'''
        
        return np.power(np.arange(N), order-1) * np.exp(-2 * np.pi * bandwidth * np.arange(N))

    def retScale(self, bw, order):
        '''Return scale as a function of bandwidth and order (samples)
            bw := spectral width (norm. freq.)
            order := order of function'''

        return 1./bw * order * np.exp(1./order)

    def retBandwidth(self, scale, order):
        '''Return bandwidth as a function of order and scale (norm. freq)
            scale := length of vector
            order := order of function'''
        
        return order * np.exp(1./order) * 1./scale

setattr(GammaGen, 'gen', atom_.gamma_)            
setattr(GammaGen, 'genFM', atom_.gammaFM_)

class FOFGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'FOF'
        self.dictionarySdifType = 'XFOF'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('rise', int), ('decay', int)]
        self.bookSdifType = 'XFOM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('rise', int), ('decay', int), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('rise', int), ('order', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp', 'rise', 'decay']
        self.winargs = ['rise', 'decay']

    def window(self, N, rise=64 , decay=128):

        t = np.arange(rise)
        
        #bandwidth(i.e. duration)
        op = np.log(decay) / decay
        
        #skirt width 
        factor = np.pi * 1./rise
        f = 0.5 * (1. - np.cos(factor * t) * np.exp(-op * t))
        p = f[rise-1]
        t = np.arange(rise, N)
        ex = np.exp(-op * t)
        ex = ex / max(abs(ex)) * p
        out = np.zeros(N)
        out[0:rise] = f
        out[rise-1 : N-1] = ex
        return out

    #here it is better to consider a symetrical extension of the rise time
    def enbw(self, N, **winargs):
        w = self.window(N, **winargs)
        w_ = w[0:winargs['rise']]
        w__ = np.concatenate((w_, w_[::-1]))
        N_ = winargs['rise']

        return N_ * np.sum(w__**2) / np.sum(w_)**2
        

    def minHop(self, N, **winargs):
        '''An estimate of the minimum hop expressed as a fraction of the window size, where:
           N := size of the window
           winargs := additional window arguments'''

        return winargs['rise'] / (2 * winargs['rise'] * self.enbw(N, **winargs))

        
#mixin
setattr(FOFGen, 'gen', atom_.fof_)
setattr(FOFGen, 'genFM', atom_.fofFM_)

class DampedGen(Window, Sinusoid):

    def __init__(self):
        self.type = 'damped'
        self.dictionarySdifType = 'XDMP'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('damp', float)]
        self.bookSdifType = 'XDMM'
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('damp', float), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('damp', float), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp', 'damp']
        self.winargs = ['damp']

    def window(self, N, damp=0.1):
        return np.exp(-damp * np.arange(N))


#mixin 
setattr(DampedGen, 'gen', atom_.damped_)
setattr(DampedGen, 'genFM', atom_.dampedFM_)

#Harmonic Types#
################

class HarmonicGaborGen(GaborGen, HarmonicSinusoid):

    def __init__(self):
        self.type = 'hGabor'
        self.dictionarySdifType = 'XGHB'
        self.dictionaryType = [('type', 'S10'), ('instrument', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int)]
        self.bookSdifType = 'XGHM'
        self.bookType = [('type', 'S10'), ('instrument', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp']
        self.winargs = []

class HarmonicHannGen(HannGen, HarmonicSinusoid):

    def __init__(self):
        self.type = 'hHann'
        self.dictionarySdifType = 'XHHB'
        self.dictionaryType = [('type', 'S10'), ('instrument', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int)]
        self.bookSdifType = 'XHHM'
        self.bookType = [('type', 'S10'), ('instrument', 'S10'), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int), ('mag', float), ('phase', float)]
        self.inputType = [('time', float), ('index', int), ('onset', int), ('duration', int), ('omega', float), ('chirp', float), ('mag', float), ('phase', float), ('k_mag', float), ('k_phase', float), ('pitch', int), ('structure', int), ('component', int), ('mag', float), ('phase', float), ('norm', float)]
        self.genargs = ['duration', 'omega', 'chirp']
        self.winargs = []

#Soundgrain#
############
class SoundgrainGen(object):

    def __init__(self):
        self.type = 'soundgrain'
        self.dictionarySdifType = 'XSGR'
        self.bookSdifType = 'XSGM'
        self.dictionaryType = [('type', 'S10'), ('onset', int), ('duration', int), ('corpus_index', int), ('file_index', int), ('norm', float)]
        self.bookType = [('type', 'S10'), ('onset', int), ('duration', int), ('corpus_index', int), ('file_index', int), ('norm', float), ('mag', float)]
        self.inputType = [('time', float), ('onset', int), ('duration', int), ('corpus_index', int), ('file_index', int), ('norm', float), ('mag', float)]
