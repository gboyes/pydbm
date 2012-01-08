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

import time
import re
import os
import copy
import itertools

import numpy as np
import scipy.fftpack as fftpack
import scipy.linalg as linalg
import scipy.interpolate as interpolate
import pysdif
import numpy.lib.recfunctions as rfn

import pydbm.meta
import pydbm.atom
import pydbm.data
import pydbm.book
import pydbm.utils

#FIX: module needs some cleanup and refactoring

#most basic dictionary#
#######################
class Dictionary(pydbm.meta.Types, pydbm.meta.Group, pydbm.utils.Utils):
    '''Time-frequency analysis dictionary'''

    def __init__(self, fs):
        pydbm.meta.Types.__init__(self)
        self.order = 1
        self.sdifType = 'XDIC'
        self.sampleRate = fs
        self.atoms = np.array([], dtype=[('index', 'i4', (1, ))])

    def index(self): 
        '''Index the set of atoms'''

        for i in xrange(len(self.atoms)):
            self.atoms['index'][i][0] = i

    def toBlockDictionary(self):
        '''Make a BlockDictionary out of a Dictionary'''

        BD = BlockDictionary(self.sampleRate)

        for dtype in np.unique(self.atoms['type']):
            winargs = self.atomGenTable[dtype].winargs
            inds = np.where(self.atoms['type'] == dtype)[0]
            v = np.unique(self.atoms[inds][[a[0] for a in self.atomGenTable[dtype].dictionaryType if a[0] not in ['omega', 'onset']]])
            vd = [p[0] for p in v.dtype.descr]

            for ti, t in enumerate(v):
                k = np.where(self.atoms[inds][vd] == t)[0]
                omegas = np.unique(self.atoms[inds][k]['omega'])
                onsets = np.unique(self.atoms[inds][k]['onset'])
                
                if winargs:
                    BD.addBlock(dtype, v[ti]['duration'], v[ti]['chirp'], omegas, onsets, **dict(zip(self.atomGenTable[dtype].winargs, v[self.atomGenTable[dtype].winargs][ti])))
                else:
                    BD.addBlock(dtype, v[ti]['duration'], v[ti]['chirp'], omegas, onsets)
                
        return BD

    #write the dictionary to an SDIF file, this should be made more general
    def writeSDIF(self, outpath):
        '''Generate an SDIF file from the dictionary
           outpath := the path of the sdif to be written'''
        
        f = pysdif.SdifFile('%s'%outpath, 'w')
        f.add_NVT({'date' : time.asctime(time.localtime()), 'sample rate' : self.sampleRate})
        f.add_frame_type(self.sdifType, 'XGAB NewXGAB, XFOF NewXFOF, XGMA NewXGMA, XDMP NewXDMP, XSGR NewXSGR')

        self.atoms.sort(order='onset')

        f.add_matrix_type('XGAB', 'onset, scale, omega, chirp')
        f.add_matrix_type('XFOF', 'onset, scale, omega, chirp, rise, decay')
        f.add_matrix_type('XGMA', 'onset, scale, omega, chirp, order, bandwidth')
        f.add_matrix_type('XDMP', 'onset, scale, omega, chirp, damp')
        f.add_matrix_type('XSGR', 'onset, corpus_index, file_index, norm')

        if any(self.atoms['type'] == 'soundgrain'):
            f.add_NVT(dict(zip([str(k) for k in range(len(self.corpus.corpora))], self.corpus.corpora)))  
        
        n = 0
        while n < self.num():
            t = self.atoms['onset'][n]
            frame = f.new_frame(self.sdifType, t /float(self.sampleRate))

            c = 0
            for ind in np.where(self.atoms['onset'] == t)[0]:
                N = self.atoms[ind]

                if N['type'] == 'gabor':
                    frame.add_matrix('XGAB', np.array([[N['onset'], N['duration'], N['omega'], N['chirp']]]))

                elif N['type'] == 'FOF':
                    frame.add_matrix('XFOF', np.array([[N['onset'], N['duration'], N['omega'], N['chirp'], N['rise'], N['decay']]]))

                elif N['type'] == 'gamma':
                    frame.add_matrix('XGMA', np.array([[N['onset'], N['duration'], N['omega'], N['chirp'], N['order'], N['bandwidth']]]))
 
                elif N['type'] == 'damped':
                    frame.add_matrix('XDMP', np.array([[N['onset'], N['duration'], N['omega'], N['chirp'], N['damp']]]))

                elif N['type'] == 'soundgrain':
                    frame.add_matrix('XSGR', np.array([[N['onset'], N['corpus_index'], N['file_index'], N['norm']]]))
                    
                c += 1
    
            frame.write()
            n += c
        
        f.close()

    #methods to add atoms according to signal and symbolic objects#
    ###############################################################

    def addRegion(self, atype, scale, minsamp, maxsamp, minomega, maxomega, sampsampling, omegasampling, **kwargs):
        '''Add a set of time-frequency atoms for a quadrant in the TF plane'''
        
        sec = (np.linspace(minsamp, maxsamp, num=sampsampling, endpoint=True)).astype(int)
        freq = np.linspace(minomega, maxomega, num=omegasampling, endpoint=True)
        
        d = np.zeros(sampsampling * omegasampling, dtype=self.atomGenTable[atype].dictionaryType)
        d['type'] = atype
        
        c = 0
        for t in sec:

            for f in freq:
                d[c]['onset'] = t
                d[c]['omega'] = f
                d[c]['duration'] = scale

                #additional arguments go here
                for key in kwargs:
                    d[key][c] = kwargs[key] 

                c += 1

        self.atoms = rfn.stack_arrays((self.atoms, d)).data

    def addPolygon(self, Poly, dtype, scale, nbins, hop, **kwargs)
        '''Add a set of atoms for a give Polygon instance''':

        Poly.getPolyHull(self.sampleRate, hop, nbins)

        d = np.zeros(len(Poly.polyHull), dtype=self.atomGenTable[dtype].dictionaryType)
        d['type'] = dtype
        for key in kwargs:
            d[key] = kwargs[key] 

        for i, h in enumerate(Poly.polyHull):
            d[i]['onset'] = h['hop'] * hop
            d[i]['omega'] = h['bin'] / float(nbins) 
            d[i]['duration'] = scale 

        self.atoms = rfn.stack_arrays((self.atoms, d)).data


    def addSTFT(self, dtype, scale, hop, tmin, tmax, **kwargs):
        '''Add a set of atoms with time-frequency locations expected with a STFT, i.e. single scale/bandwidth, where
           dtype := atom data type
           scale := size of the atoms
           tmin := minimum time in samples
           tmax := maximum time in samples
           kwargs := a dictionary of additional window arguments'''

        tpoints = np.arange(tmin, tmax, hop).astype(int)
        num = len(tpoints) * (scale/2)

        d = np.zeros(num, dtype=self.atomGenTable[dtype].dictionaryType)
        d['type'] = dtype
        d['duration'] = scale        

        c = 0
        for t in tpoints:
            for k in xrange(1, scale/2):
                om = k/float(scale)
                d[c]['onset'] = t
                d[c]['omega'] = om

                #additional arguments go here
                for key in kwargs:
                    d[c][key] = kwargs[key] 

                c += 1
 
        self.atoms = rfn.stack_arrays((self.atoms, d[0:c])).data


    def addPartial(self, dtype, partial, scale, mcentwidth, tsampling, sampling, **kwargs):
        
        d = np.zeros(np.ceil(len(partial['time'])/float(tsampling)) * sampling, dtype=self.atomGenTable[dtype].dictionaryType)
        d['type'] = dtype
        N = np.floor(partial['time'] * self.sampleRate)
        c = 0
        ind = 0
        while ind < len(N):
            
            freq = partial['frequency'][ind]
            
            for dm in (np.linspace(freq - (freq*2**(mcentwidth/1200.) - freq), freq*2**(mcentwidth/1200.), num=sampling, endpoint=True) / self.sampleRate):
                d[c]['duration'] = scale
                d[c]['omega'] = dm
                d[c]['onset'] = N[ind]     
                
                #additional arguments go here
                for key in kwargs:
                    d[c][key] = kwargs[key] 

                c += 1

            ind += tsampling

        self.atoms = rfn.stack_arrays((self.atoms, d[0:c])).data


    def addTransient(self, dtype, location, minscale, maxscale, minomega, maxomega, scalesampling, omegasampling, **kwargs):
        '''Add a set of time-frequency atoms for a transient location'''

        sca = (np.linspace(minscale, maxscale, num=scalesampling, endpoint=True)).astype(int)
        freq = np.linspace(minomega, maxomega, num=omegasampling, endpoint=True)
        
        d = np.zeros(scalesampling * omegasampling, dtype=self.atomGenTable[dtype].dictionaryType)
        d['type'] = dtype
        
        c = 0
        for t in sca:

            for f in freq:
                d[c]['duration'] = t
                d[c]['omega'] = f
                d[c]['onset'] = location

                #additional arguments go here
                for key in kwargs:
                    d[c][key] = kwargs[key] 

                c += 1

        self.atoms = rfn.stack_arrays((self.atoms, d)).data


    #Analysis Functions#
    ####################

    def descent(self, signal, invec, vecsize, maxit, upwidth, paramsd, threshd, globd):
        '''Optimize a given set of parameters associated with a elementary function until a threshold is met
           invec := an array of param. -> value pairs
           vecsize := effective sample rate for each parameter 
           maxit := the maximum number of iterations in the optimization procedure
           upwidth := width of range update
           paramsd := a param -> np.array([min_param, max_param]) pair for each parameter defining the min/max deviation from the value in invec 
           threshd := a param -> threshold pair for each parameter defining the stopping condition associated with the deviation in the update range
           globd := a param -> np.array([min_param, max_param]) pair for each parameter defining the min/max global value the parameter can take (avoid boundaries)'''

        #only optimize the parameters that are in the data type
        k = [key for key in paramsd.keys() if key in [dd[0] for dd in self.atomGenTable[invec['type']].dictionaryType]]
        paramsd = dict(zip(k, [paramsd[key] for key in k])) 
        threshd = dict(zip(k, [threshd[key] for key in k]))    

        #as a list of arrays and an array for thresh 
        plist = [np.array([invec[key]-paramsd[key][0] if invec[key]-paramsd[key][0] > globd[key][0] else globd[key][0], 
                 invec[key]+paramsd[key][1] if invec[key]+paramsd[key][1] < globd[key][1] else globd[key][1]]) for key in k]
        threshs = np.array([threshd[key] for key in k]) 

        c = 0
        while (c < maxit) and (any([abs(plist[p][0] - plist[p][1]) > threshs[p] for p in xrange(len(plist))])):

            #sample the range
            alist = dict(zip(k, [np.unique(np.linspace(min(p), max(p), num=vecsize, endpoint=True).astype(type(min(p)))) for p in plist]))

            #initialize loop
            maxc = np.zeros(np.prod([len(p) for p in alist.values()]))
            dnow = invec.copy()

            #cartesian product as a generator expression
            cart = itertools.product(*alist.values())

            for pind, param in enumerate(cart):
                for ind, val in enumerate(param):
                    dnow[k[ind]] = val

                if dnow['onset'] + dnow['duration'] >= len(signal):
                    continue

                ad = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                ads = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                ad /= linalg.norm(ad)
                ads /= linalg.norm(ads)

                maxc[pind] = np.sqrt(np.inner(ad, signal[dnow['onset']:dnow['onset']+dnow['duration']])**2 + np.inner(ads, signal[dnow['onset']:dnow['onset']+dnow['duration']])**2)
        
            coor = np.unravel_index(np.argmax(np.reshape(maxc, tuple([len(p) for p in alist.values()]))), tuple([len(p) for p in alist.values()]))
            
        
            plist = [np.array([alist.values()[p][coor[p]-upwidth] if coor[p]-upwidth >= 0 else alist.values()[p][0], 
                     alist.values()[p][coor[p]+upwidth] if coor[p]+upwidth < len(alist.values()[p]) else alist.values()[p][len(alist.values()[p])-1]]) for p in xrange(len(coor))]  
        
            c +=1
 
        return dict(zip(k, [alist.values()[p][coor[p]] for p in xrange(len(coor))]))
    
    def mp(self, signal, cmax, srr_thresh):
        '''Matching pursuit using the set of atoms in the Dictionary, where
           signal := signal to decompose
           cmax := maximum number of iterations
           srr_thresh := model-to-residual ratio stopping condition'''
  
        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))
        dtype.append(('phase', float))        
        B = pydbm.book.Book(cmax, dtype, self.sampleRate)

        #place to put model
        out = np.zeros(len(signal), dtype=float)
 
        #place to hold analysis values
        max_mag = np.zeros(len(self.atoms))
        max_phase = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:

                #protip: the first number is phase ;)
                atom1 = self.atomGenTable[self.atoms['type'][cnt]].gen(0., *[self.atoms[cnt][arg] for arg in self.atomGenTable[self.atoms['type'][cnt]].genargs])
                atom2 = self.atomGenTable[self.atoms['type'][cnt]].gen(np.pi/2, *[self.atoms[cnt][arg] for arg in self.atomGenTable[self.atoms['type'][cnt]].genargs])
                    
                a1 = np.inner(atom1, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / linalg.norm(atom1)
                a2 = np.inner(atom2, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / linalg.norm(atom2)

                max_mag[cnt] = np.sqrt(a1**2 + a2**2)
                max_phase[cnt] = np.arctan2(a2, a1)

            #get and remove maximally correlated atom
            indx = np.argmax(max_mag)

            mag = max_mag[indx]
            phase = max_phase[indx]
            atom = self.atomGenTable[self.atoms['type'][indx]].gen(phase, *[self.atoms[indx][arg] for arg in self.atomGenTable[self.atoms['type'][indx]].genargs])

            norman = linalg.norm(atom)
            atom *= 1./norman

            signal[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] -= atom * mag
            out[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] += atom * mag
            
            #Store decomposition Values
            B.atoms['type'][c_cnt] = self.atoms['type'][indx]
            B.atoms['onset'][c_cnt] = self.atoms['onset'][indx]
            B.atoms['mag'][c_cnt] = mag
            B.atoms['phase'][c_cnt] = phase
            B.atoms['index'][c_cnt] = c_cnt
            #B.atoms['norm'][c_cnt] = norman 

            #set the free args
            fixedarg = ['type', 'onset', 'mag', 'phase', 'norm']
            for param in [P[0] for P in self.atomGenTable[self.atoms[indx]['type']].dictionaryType if P[0] not in fixedarg]: 
                B.atoms[param][c_cnt] = self.atoms[indx][param]

            srr = 10 * np.log10( linalg.norm(out)**2 / linalg.norm(signal)**2 ) 
            print(mag**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)
 
            #indices to update
            up_ind = np.union1d(np.intersect1d(np.where(self.atoms['onset'] + self.atoms['duration'] < self.atoms[indx]['onset'] + self.atoms[indx]['duration'])[0], 
                                    np.where(self.atoms['onset'] + self.atoms['duration'] >= self.atoms[indx]['onset'])[0]),
                                np.intersect1d(np.where(self.atoms['onset'] < self.atoms[indx]['onset'] + self.atoms[indx]['duration'])[0], 
                                    np.where(self.atoms['onset']  >= self.atoms[indx]['onset'])[0]))
            max_mag[up_ind] = 0.

            c_cnt += 1

        B.atoms = B.atoms[0:c_cnt]
        B.model = np.array(out)
        B.residual = np.array(signal)
        B.resPercent =  (linalg.norm(B.residual)**2 / start_norm**2) * 100
        B.srr = srr

        return out, signal, B

    def mp_descent(self, signal, cmax, srr_thresh, vecsize, maxit, upwidth, paramsd, threshd, globd):

        #initialize
        dtype = self.atoms.dtype.descr
        #dtype.append(('index', int))
        dtype.append(('mag', float))
        dtype.append(('phase', float))
        #dtype.append(('norm', float))
        
        B = pydbm.book.Book(cmax, dtype, self.sampleRate)

        #place to put model
        out = np.zeros(len(signal), dtype=float)

        #place to hold analysis values
        max_mag = np.zeros(len(self.atoms))
        max_phase = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:

                atom1 = self.atomGenTable[self.atoms['type'][cnt]].gen(0., *[self.atoms[cnt][arg] for arg in self.atomGenTable[self.atoms['type'][cnt]].genargs])
                atom2 = self.atomGenTable[self.atoms['type'][cnt]].gen(np.pi/2, *[self.atoms[cnt][arg] for arg in self.atomGenTable[self.atoms['type'][cnt]].genargs])
                    
                a1 = np.inner(atom1, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / linalg.norm(atom1)
                a2 = np.inner(atom2, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / linalg.norm(atom2)

                max_mag[cnt] = np.sqrt(a1**2 + a2**2)
                max_phase[cnt] = np.arctan2(a2, a1)

            #get the parameters of the maximally correlated atom
            indx = np.argmax(max_mag)

            opt_param = self.descent(signal, self.atoms[indx].copy(), vecsize, maxit, upwidth, paramsd, threshd, globd)
            
            dnow = self.atoms[indx].copy()
            for key in opt_param.keys():
                dnow[key] = opt_param[key]

            g = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
            g /= linalg.norm(g)
            gs = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
            gs /= linalg.norm(gs)

            mag = np.sqrt(np.inner(g, signal[dnow['onset']:dnow['onset']+dnow['duration']])**2 + np.inner(gs, signal[dnow['onset']:dnow['onset']+dnow['duration']])**2)
            phase = np.arctan2(np.inner(gs, signal[dnow['onset']:dnow['onset']+dnow['duration']]), np.inner(g, signal[dnow['onset']:dnow['onset']+dnow['duration']]))

            atom = self.atomGenTable[dnow['type']].gen(phase, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])

            norman = linalg.norm(atom)
            atom *= 1./norman

            signal[dnow['onset'] : dnow['onset'] + dnow['duration']] -= atom * mag
            out[dnow['onset'] : dnow['onset'] + dnow['duration']] += atom * mag
            
            #Store decomposition Values
            B.atoms['type'][c_cnt] = dnow['type']
            B.atoms['onset'][c_cnt] = dnow['onset']
            B.atoms['mag'][c_cnt] = mag
            B.atoms['phase'][c_cnt] = phase
            #B.atoms['norm'][c_cnt] = norman 

            #set the 'variable' args
            fixedarg = ['type', 'onset', 'mag', 'phase', 'norm']
            for param in [P[0] for P in self.atomGenTable[dnow['type']].dictionaryType if P[0] not in fixedarg]: 
                B.atoms[param][c_cnt] = dnow[param] 

            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2) 
            print(mag**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)

            #indices to update
            up_ind = np.union1d(np.intersect1d(np.where(self.atoms['onset'] + self.atoms['duration'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset'] + self.atoms['duration'] >= dnow['onset'])[0]),
                                np.intersect1d(np.where(self.atoms['onset'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset']  >= dnow['onset'])[0]))
 
            max_mag[up_ind] = 0.

            c_cnt += 1

        B.atoms = B.atoms[0:c_cnt]
        B.model = np.array(out)
        B.residual = np.array(signal)
        B.resPercent =  (linalg.norm(B.residual)**2 / start_norm**2) * 100
        B.srr = srr

        return out, signal, B

#Spectral Dictionary#
#####################

class SpecDictionary(Dictionary, pydbm.meta.Spectral, pydbm.utils.Utils):

    def __init__(self, fs):
        pydbm.meta.Spectral.__init__(self)
        self.order = 2
        self.sdifType = 'XSDI'
        self.sampleRate = fs
        self.atoms = np.array([], dtype=[('index', 'int', (2, ))])


    #add a Note object
    def addNote(self, dtype, midicents, scales, onsets, **kwargs):
        
        '''Add atoms based on a f0''' 

        d = np.zeros(sum([len(o) for o in onsets]), dtype=self.atomGenTable[dtype].dictionaryType)
        d['type'] = dtype
        d['omega'] = self.midi2hz(midicents / 100) / self.sampleRate

        #additional arguments go here
        for key in kwargs:
            d[key] = kwargs[key]

        c = 0
        for si, s in enumerate(scales):
            for n in onsets[si]:
                
                d['duration'][c] = s
                d['onset'][c] = n
                c += 1
                    
        self.atoms = rfn.stack_arrays((self.atoms, d)).data


    #Analysis Functions#
    ####################

    def mp(self, signal, cmax, srr_thresh, tolmidicents, maxPeaks, dBthresh, overspec):

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))
        dtype.append(('phase', float))

        book = pydbm.book.SpectralBook(cmax * maxPeaks, dtype, self.sampleRate)

        out = np.zeros(len(signal))

        #place to hold analysis values
        max_mag = np.zeros(self.num())
        up_ind = np.arange(self.num())

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        last_srr = 1.
        srr = 0.
        c_cnt = 0
        t_cnt = 0

        tol = tolmidicents / float(100)

        while (c_cnt < cmax) and (srr <= srr_thresh) and (abs(last_srr - srr) > 0):

            print(c_cnt)

            for cnt in up_ind:

                w = self.atomGenTable[self.atoms['type'][cnt]].winargs
                if w:
                    winargs = dict(zip(w, self.atoms[w][cnt]))
                else:
                    winargs = {}
 
                win = self.atomGenTable[self.atoms['type'][cnt]].window(self.atoms['duration'][cnt], **winargs)
                X = fftpack.fft(signal[self.atoms['onset'][cnt] : self.atoms['onset'][cnt] + self.atoms['duration'][cnt]] * win)
 
                L, V = self.pickPeaks(abs(X), maxPeaks, dBthresh, self.atomGenTable[self.atoms['type'][cnt]].enbw(self.atoms['duration'][cnt], **winargs), self.atoms['duration'][cnt]*overspec, self.atoms['type'][cnt], **winargs)

                if len(L) == 0:
                   max_mag[cnt] = -np.inf
                   continue

                L, V, P = self.interpolateValues(abs(X), np.angle(X), self.atoms['duration'][cnt]*overspec, L, V, self.atoms['duration'][cnt], self.atoms['type'][cnt], **winargs)

                #sort according to ascending frequency
                sind = np.argsort(L)
                L = L[sind]

                #remove anything below the 'fundamental', make this logarithmic...
                L = L[(L != 0.) & (self.hz2midi(L / self.atoms['duration'][cnt] * self.sampleRate) > self.hz2midi(self.atoms['omega'][cnt] * self.sampleRate) - tol)]

                if (len(L) == 0):
                    max_mag[cnt] = -np.inf
                    continue

                #fundamental
                f0 = L[np.argmin(abs(L / self.atoms['duration'][cnt] - self.atoms['omega'][cnt]))] / self.atoms['duration'][cnt]
                dist = abs(self.hz2midi(f0 * self.sampleRate) - self.hz2midi(self.atoms['omega'][cnt] * self.sampleRate))

                if dist >= tol:
                    max_mag[cnt] = -np.inf
                    continue

                L = L[np.where(L / self.atoms['duration'][cnt] >= f0)[0]]
            
                fnote = self.hz2midi(L/self.atoms['duration'][cnt] * self.sampleRate)
            
                #theoretical note locations
                tnote = self.hz2midi(f0 * np.arange(1, maxPeaks) * self.sampleRate)

                inds = [i for (i, val) in enumerate(fnote) if np.min(abs(tnote - val)) <= tol] 

                if len(inds) > 0:

                    phi = np.zeros(len(inds))
                    mags = np.zeros(len(inds))
                    dnow = self.atoms[cnt].copy()

                    #get phases and magnitudes
                    for ii, i in enumerate(inds):
                        dnow['omega'] = L[i] / dnow['duration']
                        a1 = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                        a1 /= linalg.norm(a1)
                        a2 = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                        a2 /= linalg.norm(a2)
                 
                        c1 = np.inner(a1, signal[dnow['onset'] : dnow['onset'] + dnow['duration']])
                        c2 = np.inner(a2, signal[dnow['onset'] : dnow['onset'] + dnow['duration']])
                 
                        phi[ii] = np.arctan2(c2, c1) 
                        mags[ii] = np.sqrt(c1**2 + c2**2)

                    max_mag[cnt] = np.sum(mags)
                
            ind = np.argmax(max_mag)
            dnow = self.atoms[ind].copy()

            w = self.atomGenTable[self.atoms['type'][ind]].winargs
            if w : 
                winargs = dict(zip(w, self.atoms[w][ind]))
            else:
                winargs = {}

            win = self.atomGenTable[dnow['type'] ].window(dnow['duration'], **winargs)
            X = fftpack.fft(signal[dnow['onset'] : dnow['onset']  + dnow['duration']] * win) 

            L, V = self.pickPeaks(abs(X), maxPeaks, dBthresh, self.atomGenTable[dnow['type']].enbw(dnow['duration'], **winargs), dnow['duration']*overspec, dnow['type'], **winargs)    
            L, V, P = self.interpolateValues(abs(X), np.angle(X), dnow['duration']*overspec, L, V, dnow['duration'] , dnow['type'] , **winargs)

            #sort according to ascending frequency
            sind = np.argsort(L)
            L = L[sind]

            #remove anything below the 'fundamental', make this logarithmic...
            L = L[(L != 0.) & (self.hz2midi(L / dnow['duration'] * self.sampleRate) > self.hz2midi(dnow['omega'] * self.sampleRate) - tol)]
             
            if len(L) == 0:
                last_srr = srr
                continue
            
            #fundamental
            f0 = L[np.argmin(abs(L / dnow['duration'] - dnow['omega']))] / dnow['duration']
            dist = abs(self.hz2midi(f0 * self.sampleRate) - self.hz2midi(dnow['omega'] * self.sampleRate))

            if dist >= tol:
                last_srr = srr
                continue

            L = L[np.where(L / dnow['duration'] >= f0)[0]]
            
            fnote = self.hz2midi(L/dnow['duration'] * self.sampleRate)
            
            #theoretical note locations
            tnote = self.hz2midi(f0 * np.arange(1, maxPeaks) * self.sampleRate)
            inds = [i for (i, val) in enumerate(fnote) if np.min(abs(tnote - val)) <= tol] 

            if len(inds) == 0:
                last_srr = srr
                continue

            atom = np.zeros(dnow['duration'])

            #get phases and magnitudes and subtract (no harmonicity assumption so the atoms might not be mutually orthogonal, min distance means they should be roughly so though)
            for ii, i in enumerate(inds):
                dnow['omega'] = L[i] / dnow['duration'] 
                a1 = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                a1 /= linalg.norm(a1)
                a2 = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                a2 /= linalg.norm(a2)
                c1 = np.inner(a1, signal[dnow['onset'] : dnow['onset'] + dnow['duration']])
                c2 = np.inner(a2, signal[dnow['onset'] : dnow['onset'] + dnow['duration']])
                phi = np.arctan2(c2, c1) 
                mag = np.sqrt(c1**2 + c2**2)

                a1 = self.atomGenTable[dnow['type']].gen(phi, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                a1 /= linalg.norm(a1)

                book.atoms[t_cnt]['type'] = dnow['type']
                book.atoms[t_cnt]['onset'] = dnow['onset']
                book.atoms[t_cnt]['duration'] = dnow['duration']
                book.atoms[t_cnt]['omega'] = dnow['omega']
                book.atoms[t_cnt]['mag'] = mag
                book.atoms[t_cnt]['phase'] = phi
                book.atoms[t_cnt]['index'][0] = c_cnt #structure
                book.atoms[t_cnt]['index'][1] = ii #component

                for key in winargs.keys():
                    book.atoms[t_cnt][key] = dnow[key]

                t_cnt += 1

                signal[dnow['onset'] : dnow['onset'] + dnow['duration']] -= a1 * mag
                out[dnow['onset'] : dnow['onset'] + dnow['duration']] += a1 * mag


            last_srr = srr
            srr = 10 * np.log10( linalg.norm(out)**2 / linalg.norm(signal)**2 ) 
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)
 
            #indices to update
            up_ind = np.union1d(np.intersect1d(np.where(self.atoms['onset'] + self.atoms['duration'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset'] + self.atoms['duration'] >= dnow['onset'])[0]),
                                np.intersect1d(np.where(self.atoms['onset'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset']  >= dnow['onset'])[0]))
            max_mag[up_ind] = 0.
            

            c_cnt += 1

        book.atoms = book.atoms[0:t_cnt]
        book.model = np.array(out)
        book.residual = np.array(signal)
        book.resPercent =  (linalg.norm(book.residual)**2 / start_norm**2) * 100
        book.srr = srr

        return out, signal, book

    def mp2(self, signal, cmax, srr_thresh, tolmidicents, maxPeaks, dBthresh, overspec):

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))
        dtype.append(('phase', float))
        book = pydbm.book.SpectralBook(cmax * maxPeaks, dtype, self.sampleRate)
        out = np.zeros(len(signal))

        #place to hold analysis values
        max_mag = np.zeros(self.num())
        up_ind = np.arange(self.num())

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        last_srr = 1.
        srr = 0.
        c_cnt = 0
        t_cnt = 0

        tol = tolmidicents / float(100)

        while (c_cnt < cmax) and (srr <= srr_thresh) and (abs(last_srr - srr) > 0):
            print(c_cnt)
            for cnt in up_ind:
                w = self.atomGenTable[self.atoms['type'][cnt]].winargs
                if w:
                    winargs = dict(zip(w, self.atoms[w][cnt]))
                else:
                    winargs = {}
                win = self.atomGenTable[self.atoms['type'][cnt]].window(self.atoms['duration'][cnt], **winargs)
                X = fftpack.fft(signal[self.atoms['onset'][cnt] : self.atoms['onset'][cnt] + self.atoms['duration'][cnt]] * win)
                L, V = self.pickPeaks(abs(X), maxPeaks, dBthresh, self.atomGenTable[self.atoms['type'][cnt]].enbw(self.atoms['duration'][cnt], **winargs), self.atoms['duration'][cnt]*overspec, self.atoms['type'][cnt], **winargs)

                if len(L) == 0:
                   max_mag[cnt] = -np.inf
                   continue

                L, V, P = self.interpolateValues(abs(X), np.angle(X), self.atoms['duration'][cnt]*overspec, L, V, self.atoms['duration'][cnt], self.atoms['type'][cnt], **winargs)

                #sort according to ascending frequency
                sind = np.argsort(L)
                L = L[sind]
                #remove anything below the 'fundamental', make this logarithmic...
                L = L[(L != 0.) & (self.hz2midi(L / self.atoms['duration'][cnt] * self.sampleRate) > self.hz2midi(self.atoms['omega'][cnt] * self.sampleRate) - tol)]
                if (len(L) == 0):
                    max_mag[cnt] = -np.inf
                    continue

                #fundamental
                f0 = L[np.argmin(abs(L / self.atoms['duration'][cnt] - self.atoms['omega'][cnt]))] / self.atoms['duration'][cnt]
                dist = abs(self.hz2midi(f0 * self.sampleRate) - self.hz2midi(self.atoms['omega'][cnt] * self.sampleRate))

                if dist >= tol:
                    max_mag[cnt] = -np.inf
                    continue

                L = L[np.where(L / self.atoms['duration'][cnt] >= f0)[0]]
                fnote = self.hz2midi(L/self.atoms['duration'][cnt] * self.sampleRate)
            
                #theoretical note locations
                tnote = self.hz2midi(f0 * np.arange(1, maxPeaks) * self.sampleRate)
                inds = [i for (i, val) in enumerate(fnote) if np.min(abs(tnote - val)) <= tol] 

                if len(inds) > 0:
                    dnow = self.atoms[cnt].copy()
                    pmag = np.zeros((dnow['duration'], len(inds)))
                    pphase = pmag.copy()
                    xnow = signal[dnow['onset'] : dnow['onset'] + dnow['duration']]
                    
                    for ii, i in enumerate(inds):
                        dnow['omega'] = L[i] / dnow['duration']
                        a1 = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                        a1 *= (1./linalg.norm(a1))
                        a2 = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                        a2 *= (1./linalg.norm(a2))
                        pmag[:, ii] = a1 
                        pphase[:, ii] = a2

                    lsq_m = linalg.lstsq(pmag, xnow)[0] 
                    lsq_p = linalg.lstsq(pphase, xnow)[0]
                    phi = np.arctan2(lsq_p, lsq_m)
                    mags = np.sqrt(lsq_m**2 + lsq_p**2)
                    max_mag[cnt] = np.sum(mags)
                
            ind = np.argmax(max_mag) 
            dnow = self.atoms[ind].copy()

            w = self.atomGenTable[self.atoms['type'][ind]].winargs
            if w : 
                winargs = dict(zip(w, self.atoms[w][ind]))
            else:
                winargs = {}

            win = self.atomGenTable[dnow['type'] ].window(dnow['duration'], **winargs)
            X = fftpack.fft(signal[dnow['onset'] : dnow['onset']  + dnow['duration']] * win) 

            L, V = self.pickPeaks(abs(X), maxPeaks, dBthresh, self.atomGenTable[dnow['type']].enbw(dnow['duration'], **winargs), dnow['duration']*overspec, dnow['type'], **winargs)    
            L, V, P = self.interpolateValues(abs(X), np.angle(X), dnow['duration']*overspec, L, V, dnow['duration'] , dnow['type'] , **winargs)

            #sort according to ascending frequency
            sind = np.argsort(L)
            L = L[sind]

            #remove anything below the 'fundamental', make this logarithmic...
            L = L[(L != 0.) & (self.hz2midi(L / dnow['duration'] * self.sampleRate) > self.hz2midi(dnow['omega'] * self.sampleRate) - tol)] 
            if len(L) == 0:
                last_srr = srr
                continue
            
            #fundamental
            f0 = L[np.argmin(abs(L / dnow['duration'] - dnow['omega']))] / dnow['duration']
            dist = abs(self.hz2midi(f0 * self.sampleRate) - self.hz2midi(dnow['omega'] * self.sampleRate))
            if dist >= tol:
                last_srr = srr
                continue

            L = L[np.where(L / dnow['duration'] >= f0)[0]]
            fnote = self.hz2midi(L/dnow['duration'] * self.sampleRate)
            
            #theoretical note locations
            tnote = self.hz2midi(f0 * np.arange(1, maxPeaks) * self.sampleRate)
            inds = [i for (i, val) in enumerate(fnote) if np.min(abs(tnote - val)) <= tol] 
            if len(inds) == 0:
                last_srr = srr
                continue

            xnow = signal[dnow['onset'] : dnow['onset'] + dnow['duration']]
            pmag = np.zeros((dnow['duration'], len(inds)))
            pphase = pmag.copy()

            for ii, i in enumerate(inds):
                dnow['omega'] = L[i] / dnow['duration']
                a1 = self.atomGenTable[dnow['type']].gen(0., *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                a1 *= (1./linalg.norm(a1))
                a2 = self.atomGenTable[dnow['type']].gen(np.pi/2, *[dnow[arg] for arg in self.atomGenTable[dnow['type']].genargs])
                a2 *= (1./linalg.norm(a2))
                pmag[:, ii] = a1 
                pphase[:, ii] = a2 
                book.atoms[t_cnt]['omega'] = dnow['omega']
                book.atoms[t_cnt]['type'] = dnow['type']
                book.atoms[t_cnt]['onset'] = dnow['onset']
                book.atoms[t_cnt]['duration'] = dnow['duration']
                book.atoms[t_cnt]['index'][1] = ii #component
                book.atoms[t_cnt]['index'][0] = c_cnt
                t_cnt += 1
                
            lsq_m = linalg.lstsq(pmag, xnow)[0] 
            lsq_p = linalg.lstsq(pphase, xnow)[0]
            phi = np.arctan2(lsq_p, lsq_m)
            mags = np.sqrt(lsq_m**2 + lsq_p**2)

            atom = np.hstack(np.array((np.matrix(pmag) * np.vstack(lsq_m)))) + np.hstack(np.array((np.matrix(pphase) * np.vstack(lsq_p))))

            signal[dnow['onset'] : dnow['onset'] + dnow['duration']] -= atom
            out[dnow['onset'] : dnow['onset'] + dnow['duration']] += atom

            book.atoms[t_cnt-len(inds):t_cnt]['mag'] = mags
            book.atoms[t_cnt-len(inds):t_cnt]['phase'] = phi
            
            for key in winargs.keys():
                book.atoms[t_cnt-len(inds):t_cnt][key] = dnow[key]
            
            last_srr = srr
            srr = 10 * np.log10( linalg.norm(out)**2 / linalg.norm(signal)**2 ) 
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)
 
            #indices to update
            up_ind = np.union1d(np.intersect1d(np.where(self.atoms['onset'] + self.atoms['duration'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset'] + self.atoms['duration'] >= dnow['onset'])[0]),
                                np.intersect1d(np.where(self.atoms['onset'] < dnow['onset'] + dnow['duration'])[0], 
                                    np.where(self.atoms['onset']  >= dnow['onset'])[0]))
            max_mag[up_ind] = 0.
            

            c_cnt += 1

        book.atoms = book.atoms[0:t_cnt]
        book.model = np.array(out)
        book.residual = np.array(signal)
        book.resPercent =  (linalg.norm(book.residual)**2 / start_norm**2) * 100
        book.srr = srr

        return out, signal, book

    
#Instrument-Specific Dictionaries#
##################################
#FIX or scrap
class InstrumentDictionary(pydbm.data.InstrumentSubspace, Dictionary):

    '''Class to build a dictionary of instrument-specific atoms with pursuits that consider these structures'''

    def __init__(self, fs, inst):
        self.order = 2
        self.sampleRate = fs
        self.instrument = inst
        self.atoms = np.array([], dtype=[('index', '2int')])

    #make an add subspace method, for decomp from spectral MP
    def addPitch(self, pitchcents, dtype, scale, timearray, **kwargs):
        '''Add a set of sturctured harmonic atoms for a given pitch'''        

        M = np.array([self.midi2hz(k /100.) / float(self.sampleRate) for k in xrange(pitchcents-tolcents, pitchcents+tolcents, centssampling)])
        A = np.zeros(len(M) * len(timearray) * sum([len(p) for p in self.partialAmplitudes[pitchcents] if len(p) >= mincomponents and len(p) <= maxcomponents]), 
dtype=self.atomGenTable[dtype].dictionaryType)
        
        A['type'] = dtype
        A['duration'] = scale
 
        c = 0
        cc = 0
        for t in timearray:

            for mi, m in enumerate(M):

                for array_i, array in enumerate(self.partialAmplitudes[pitchcents]):
                    if len(array) >= mincomponents and len(array) <= maxcomponents:
                        f0 = self.partialFrequencies[pitchcents][array_i][0]
                        for a_i, a in enumerate(array):

                            A['k_mag'][c] = a
                            A['k_phase'][c] = self.partialPhases[pitchcents][array_i][a_i] #this just sets a value, the phase will have to be determined in the pursuit
                            A['omega'][c] = m * (self.partialFrequencies[pitchcents][array_i][a_i] / f0)
                            A['index'][c][1] =  cc
                            A['index'][c][0] = a_i
                            A['onset'][c] = t
                            A['pitch'][c] = self.hz2midi(m * self.sampleRate) * 100
     
                            #additional arguments go here
                            for key in kwargs:
                                A[key][c] = kwargs[key]

                            c += 1

                        cc += 1

        self.atoms = A
    
    def normalizeStructures(self):
 
        '''Normalize the mags of each structure so that they have equal energy regardless of scale and no. of components, i.e. norm structured atom = 1.'''        
 
        for i in np.unique(self.atoms['index'][1]):
            inds = np.where(self.atoms['index'][1] == i)[0]
            self.atoms[inds]['k_mag'] *= 1./np.sqrt(np.sum(self.atoms[inds]['k_mag']**2))


    def mp(self, signal, cmax, srr_thresh):
        '''Matching pursuit designed to use a dictionary of instrument-specific harmonic atoms, where
           signal := signal to decompose
           cmax := maximum number of iterations
           srr_thresh := model-to-residual ratio threshold breaking condition'''

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))
        dtype.append(('phase', float))
        dtype.append(('norm', float))
        
        B = pydbm.book.Book(cmax * np.max(self.atoms['index'][0]) + 1, dtype, self.sampleRate)

        #place to put model
        out = np.zeros(len(signal))

        #place to hold analysis values
        f0i = np.where(self.atoms['index'][0] == 0)[0] #we only really need to search the f0s, since everything is built from there 
        max_mag = np.zeros(len(f0i))
        max_phase = max_mag.copy()
        up_ind = np.arange(len(f0i))

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0
        t_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:
                
                di = f0i[cnt]

                compi = np.intersect1d( np.where(self.atoms['source'] == self.atoms[di]['source'])[0], np.where(self.atoms['index'][1] == self.atoms[di]['index'][1])[0]) 
                a_1 = np.zeros(self.atoms[di]['duration'])
                a_2 = np.zeros(self.atoms[di]['duration'])
                
                for cind in compi:

                    a__1 = self.atomGenTable[self.atoms[cind]['type']].gen(0., *[self.atoms[cind][arg] for arg in self.atomGenTable[self.atoms[cind]['type']].genargs])
                    a__1 /= linalg.norm(a__1)
                    c__1 = np.inner(a__1, signal[self.atoms[cind]['onset'] : self.atoms[cind]['onset'] + self.atoms[cind]['duration']])

                    a__2 = self.atomGenTable[self.atoms[cind]['type']].gen(np.pi/2, *[self.atoms[cind][arg] for arg in self.atomGenTable[self.atoms[cind]['type']].genargs])
                    a__2 /= linalg.norm(a__2)
                    c__2 = np.inner(a__2, signal[self.atoms[cind]['onset'] : self.atoms[cind]['onset'] + self.atoms[cind]['duration']])

                    self.atoms[cind]['k_phase'] = np.arctan2(c__2, c__1) #overwrite the phase value to get it back later
                     
                    a__1 = self.atomGenTable[self.atoms[cind]['type']].gen(self.atoms[cind]['k_phase'], *[self.atoms[cind][arg] for arg in self.atomGenTable[self.atoms[cind]['type']].genargs])
                    a__1 /= linalg.norm(a__1)
                    a_1 += a__1 * self.atoms[cind]['k_mag']

                    a__2 = self.atomGenTable[self.atoms[cind]['type']].gen(self.atoms[cind]['k_phase'] + np.pi/2, *[self.atoms[cind][arg] for arg in self.atomGenTable[self.atoms[cind]['type']].genargs])
                    a__2 /= linalg.norm(a__2)
                    a_2 += a__2 * self.atoms[cind]['k_mag']

                c1 = np.inner(a_1, signal[self.atoms[di]['onset'] : self.atoms[di]['onset'] + self.atoms[di]['duration']])
                c2 = np.inner(a_2, signal[self.atoms[di]['onset'] : self.atoms[di]['onset'] + self.atoms[di]['duration']])
                max_mag[cnt] = np.sqrt(c1**2 + c2**2)
                max_phase[cnt] = np.arctan2(c2, c1)

            #Globally the best
            indx = np.argmax(max_mag)
            mag = max_mag[indx]
            phase = max_phase[indx]
            
            di = f0i[indx]
            compi = np.intersect1d( np.where(self.atoms['source'] == self.atoms[di]['source'])[0], np.where(self.atoms['index'][1] == self.atoms[di]['index'][1])[0])
            atom = np.zeros(self.atoms[di]['duration'])

            for cindi, cind in enumerate(compi):

                #build the atom
                a = self.atomGenTable[self.atoms[cind]['type']].gen(self.atoms[cind]['k_phase'] + phase, *[self.atoms[cind][arg] for arg in self.atomGenTable[self.atoms[cind]['type']].genargs])
                norman = linalg.norm(a)
                a /= norman
                atom += a * self.atoms[cind]['k_mag']

                #add the parameters to the book
                B.atoms['type'][t_cnt] = self.atoms[cind]['type']
                B.atoms['source'][t_cnt] = self.atoms[cind]['source']
                B.atoms['pitch'][t_cnt] = self.atoms[cind]['pitch']
                B.atoms['onset'][t_cnt] = self.atoms[cind]['onset'] 
                B.atoms['k_mag'][t_cnt] = self.atoms[cind]['k_mag']
                B.atoms['k_phase'][t_cnt] = self.atoms[cind]['k_phase']
                B.atoms['index'][1][t_cnt] = c_cnt
                B.atoms['index'][0][t_cnt] = cindi
                B.atoms['mag'][t_cnt] = mag
                B.atoms['phase'][t_cnt] = phase
                B.atoms['norm'][t_cnt] = norman 

                #the free args
                for param in self.atomGenTable[self.atoms[cind]['type']].genargs:
                    B.atoms[param][t_cnt] = self.atoms[cind][param] 
            
                t_cnt += 1


            signal[self.atoms[di]['onset'] : self.atoms[di]['onset'] + self.atoms[di]['duration']] -= atom * mag 
            out[self.atoms[di]['onset'] : self.atoms[di]['onset'] + self.atoms[di]['duration']] += atom * mag

            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2) 
            print(mag**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)

            #indices to update
            up_ind = np.union1d(np.intersect1d(np.where(self.atoms['onset'][f0i] + self.atoms['duration'][f0i] < self.atoms[di]['onset'] + self.atoms[di]['duration'])[0], np.where(self.atoms['onset'][f0i] + self.atoms['duration'][f0i] >= self.atoms[di]['onset'])[0]),
                                np.intersect1d(np.where(self.atoms['onset'][f0i] < self.atoms[di]['onset'] + self.atoms[di]['duration'])[0], 
                                    np.where(self.atoms['onset'][f0i]  >= self.atoms[di]['onset'])[0]))
 
            max_mag[up_ind] = 0.

            c_cnt += 1

        B.atoms = B.atoms[0:t_cnt]
        B.model = np.array(out)
        B.residual = np.array(signal)
        B.resPercent =  linalg.norm(B.residual)**2 / start_norm**2 * 100
        B.srr = srr

        return out, signal, B

#Block Dictionaries#
####################
#remove this, add the functionality to regular dictionary?

class Block(pydbm.meta.Types):
    '''A high-level object describing a set of atoms with homogenous window parameters and varying time-frequency support defined by onsets and omegas'''
 
    def __init__(self, dtype, scale, chirp, omegas, onsets, **winargs):
        pydbm.meta.Types.__init__(self)

        self.dtype = dtype
        self.scale = scale
        self.chirp = chirp
        self.omegas = omegas
        self.onsets = onsets

        if not winargs:
            self.winargs = {}

        else:
            self.winargs = winargs

        for key in winargs:
            setattr(self, key, winargs[key])

        self.AtomGen = self.atomGenTable[dtype]
        self.genargs = [winargs[key] for key in [k for k in self.AtomGen.genargs if k not in ['duration', 'omega', 'chirp']]]
        self.gen = lambda phi, omega : self.AtomGen.gen(phi, self.scale, omega, chirp, *self.genargs)

class BlockDictionary(pydbm.meta.Types, pydbm.utils.Utils):

    '''a 'higher-level' setting of a dictionary ''' 

    def __init__(self, fs):
        pydbm.meta.Types.__init__(self)
        self.sampleRate = fs
        self.blocks = []
        self.dtype = []

    def addBlock(self, dtype, scale, chirp, omegas, onsets, **winargs):
        
        self.blocks.append(self.Block(dtype, scale, chirp, omegas, onsets, **winargs))

    def addPartialBlock(self, Partial, foversampfact, dtype, scale, chirp, **winargs):
        
        x = np.arange(len(Partial.array['time']))
        t = interpolate.interp1d(x, np.floor(Partial.array['time'] * self.sampleRate).astype(np.int)) 
        f = interpolate.interp1d(x, Partial.array['frequency'] /self.sampleRate)
        newx = np.linspace(0., len(Partial.array['time'])-1, len(Partial.array['time']) * foversampfact, endpoint=False)
    
        self.blocks.append(Block(dtype, scale, chirp, np.unique(f(newx)), t(newx).astype(np.int), **winargs))

    def num(self):
        if not self.blocks:
            return 0
        else:
            return sum([len(b.omegas)*len(b.onsets) for b in self.blocks]) 

    def getDtype(self):
        '''Get the effective dictionary type as the union of all block types'''

        for b in self.blocks:
            for field in b.AtomGen.dictionaryType:
                if field not in self.dtype:
                    self.dtype.append(field)
        
        return self.dtype

    def toDictionary(self):
        '''Make a Dictionary from a BlockDictionary'''
        
        D = Dictionary(self.sampleRate)
        D.atoms = np.zeros(self.num(), dtype=self.getDtype())
        c = 0
        for b in self.blocks:
            for o in b.omegas:
                for t in b.onsets:
                    D.atoms[c]['omega'] = o
                    D.atoms[c]['onset'] = t
                    D.atoms[c]['type'] = b.dtype
                    D.atoms[c]['duration'] = b.scale
                    D.atoms[c]['chirp'] = b.chirp
                    for k in b.winargs.keys():
                        D.atoms[c][k] = b.winargs[k]
                    c += 1                
        return D
    
    #Analysis functions#
    ####################

    def mp(self, signal, cmax, srr_thresh):
        '''Matching pursuit with inner product computation based on FFT, using a set of Block objects where:
           signal := signal to be decomposed
           cmax := maximum number of iterations
           srr_thresh := model-to-residual ratio'''

        #initialize
        dtype = copy.deepcopy(self.getDtype())
        #dtype.append(('index', int))
        dtype.append(('mag', float))
        dtype.append(('phase', float))
        #dtype.append(('norm', float))
        
        #the synthesis book
        book = pydbm.book.Book(cmax, dtype, self.sampleRate)
        
        #place to put model
        out = np.zeros(len(signal), dtype=float)

        #place to hold analysis values
        mags = [np.zeros((len(b.omegas), len(b.onsets))) for b in self.blocks]
        phases = copy.deepcopy(mags)
        up_inds = [np.arange(len(b.onsets)) for b in self.blocks]

        #initialize breaking condition
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):
            print(c_cnt)
            
            #different blocks could be computed in parallel as well, taking the best one at each stage
            for ib, block in enumerate(self.blocks):
                
                #if there is nothing to update for the block, there could be a dummt index for this as well
                if len(up_inds[ib]) == 0:
                    continue
 
                Nfft = 2**self.nextpow2(block.scale + ( block.onsets[up_inds[ib][-1]] - block.onsets[up_inds[ib][0]])) #a large fft
                S = fftpack.fft(signal[block.onsets[up_inds[ib][0]]  : block.onsets[up_inds[ib][-1]]], Nfft)                 

                for iom, omega in enumerate(block.omegas):
                    
                    #still a substantial bottle neck, here many values are computed that aren't strictly needed, is it possible to evaluate the crosscorrelation for specific lags? 
                    a_c = block.gen(0., omega)
                    a_c /= linalg.norm(a_c)

                    a_s = block.gen(np.pi/2, omega)
                    a_s /= linalg.norm(a_s)

                    cc = np.real(fftpack.ifft( np.conj( fftpack.fft(a_c, Nfft) ) * S)[block.onsets[up_inds[ib]] - min(block.onsets[up_inds[ib]])])
                    cs = np.real(fftpack.ifft( np.conj( fftpack.fft(a_s, Nfft) ) * S)[block.onsets[up_inds[ib]] - min(block.onsets[up_inds[ib]])])

                    mags[ib][iom, up_inds[ib]] = np.sqrt(cc**2 + cs**2)
                    phases[ib][iom, up_inds[ib]] = np.arctan2(cs, cc)

            # get the maximally correlated atom
            nm = [np.max(b) for b in mags] 
            nmag = max(nm)
            nb = np.argmax(nm)
            nind = np.unravel_index(np.argmax(mags[nb]), np.shape(mags[nb]))
            nphase = phases[nb][nind]

            atom = self.blocks[nb].gen(nphase, self.blocks[nb].omegas[nind[0]])
            atom /= linalg.norm(atom)

            signal[self.blocks[nb].onsets[nind[1]] : self.blocks[nb].onsets[nind[1]] + self.blocks[nb].scale] -= atom * nmag 
            out[self.blocks[nb].onsets[nind[1]] : self.blocks[nb].onsets[nind[1]] + self.blocks[nb].scale] += atom  * nmag

            #parameters common to all time-frequency atoms
            book.atoms[c_cnt]['type'] = self.blocks[nb].dtype
            book.atoms[c_cnt]['duration'] = self.blocks[nb].scale
            book.atoms[c_cnt]['onset'] = self.blocks[nb].onsets[nind[1]]
            book.atoms[c_cnt]['omega'] = self.blocks[nb].omegas[nind[0]]
            book.atoms[c_cnt]['chirp'] = self.blocks[nb].chirp
            book.atoms[c_cnt]['mag'] = nmag
            book.atoms[c_cnt]['phase'] = nphase

            #additional window arguments
            for key in self.blocks[nb].winargs:
                book.atoms[c_cnt][key] = self.blocks[nb].winargs[key] 

            #compute the model-residual ratio and other stats
            srr = 10 * np.log10( linalg.norm(out)**2 / linalg.norm(signal)**2 ) 
            print(nmag**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)  
            
            #fix the update interval
            for ui, u in enumerate(up_inds):
                up_inds[ui] = np.union1d(np.intersect1d(np.where(self.blocks[ui].onsets + self.blocks[ui].scale < self.blocks[nb].onsets[nind[1]] + self.blocks[nb].scale)[0], 
                                    np.where(self.blocks[ui].onsets + self.blocks[ui].scale >= self.blocks[nb].onsets[nind[1]])[0]),
                                np.intersect1d(np.where(self.blocks[ui].onsets < self.blocks[nb].onsets[nind[1]] + self.blocks[nb].scale)[0], 
                                    np.where(self.blocks[ui].onsets  >= self.blocks[nb].onsets[nind[1]])[0]))
                mags[ui][0:, up_inds[ui]] = 0.
 

            c_cnt += 1

        #add the stats to the book object
        book.atoms = book.atoms[0:c_cnt]
        book.model = np.array(out)
        book.residual = np.array(signal)
        book.resPercent =  (linalg.norm(book.residual)**2 / start_norm**2) * 100
        book.srr = srr

        return out, signal, book
        
class SoundgrainDictionary(pydbm.meta.Group, pydbm.meta.IO, pydbm.utils.Utils):

    '''Dictionary class for corpus-based synthesis'''

    def __init__(self, fs, SoundDatabase):
        pydbm.meta.IO.__init__(self)

        self.sampleRate = fs
        self.SoundDatabase = SoundDatabase
        self.atoms = np.array([], dtype=[('index', 'i4', (1, ))])

    def count(self):
        for i in xrange(self.num()):
            self.atoms['index'][i] = i

    #a pruning method based on a polygon
    def addPolygon(self, signal, win, Poly, nbins, hop, nbest):

        Poly.getPolyHull(self.sampleRate, hop, nbins)
        cg = 0.49951171875
        onset = min(Poly.polyHull['hop']) * hop
        dtype = pydbm.atom.SoundgrainGen().dictionaryType

        X = self.stft(signal, win, nbins, hop, w='hann')
        R = X[Poly.polyHull['bin'], Poly.polyHull['hop']]
        R_ = 2. * abs(R) / win / cg

        X_rms = np.sqrt( np.sum((2. * abs(X[0:nbins/2, 0:]) / win / cg)**2))
        Xsub_rms = np.sqrt( np.sum((2. * abs(X[0:nbins/2, np.unique(Poly.polyHull['hop'])]) / win / cg)**2))
        R_rms = np.sqrt( np.sum(R_**2))
        locperc = R_rms / Xsub_rms * 100

        for cind, C in enumerate(self.SoundDatabase.corpora):

            z = np.zeros(len(C.soundfiles))
            select = np.array([])

            for ind, name in enumerate(C.soundfiles):

                y, fsa = self.readAudio(C.directory + '/' + name)
                Y = self.stft(y, win, nbins, hop, w='hann')

                hull = Poly.polyHull[np.where(Poly.polyHull['hop'] - min(Poly.polyHull['hop']) < np.shape(Y)[1])[0]]
                Ry = Y[hull['bin'], hull['hop'] - min(hull['hop'])] 

                Ysub_rms = np.sqrt( np.sum((2. * abs(Y[0:nbins/2, np.unique(hull['hop']) - min(hull['hop'])]) / win / cg)**2))
                Ry_ = 2. * abs(Ry) / win / cg
                Ry_rms = np.sqrt( np.sum(Ry_**2)) 
                locpercy = Ry_rms / Ysub_rms * 100
                z[ind] = locpercy

            inds = np.argsort(z)[::-1]
            select = np.unique(np.union1d(select, inds[0:nbest]))

            #now add the approporate files to the dictionary
            da = np.zeros(len(select), dtype)
            da['type'] = 'soundgrain'

            p = 0
            for i in select:
                da['corpus_index'][p] = cind
                da['file_index'][p] = C.fileDescriptors['file_index'][i]
                da['onset'][p] = onset
                da['duration'][p] = C.fileDescriptors['length'][i]
                da['norm'][p] = C.fileDescriptors['norm'][i]

                p += 1

            self.atoms = rfn.stack_arrays((self.atoms, da)).data

        self.count()

    def addCorpus(self, onsets, corpus_index):
        '''add a Corpus to the dictonary at specific onsets'''

        N = np.array(onsets)
        dtype = pydbm.atom.SoundgrainGen().dictionaryType
        da = np.zeros(self.SoundDatabase.corpora[corpus_index].num() * len(N), dtype)
        da['type'] = 'soundgrain'

        p = 0
        for n in N:

            for i in xrange(self.SoundDatabase.corpora[corpus_index].num()):

                da['corpus_index'][p] = corpus_index
                da['file_index'][p] = self.SoundDatabase.corpora[corpus_index].fileDescriptors['file_index'][i]
                da['onset'][p] = n
                da['duration'][p] = self.SoundDatabase.corpora[corpus_index].fileDescriptors['length'][i]
                da['norm'][p] = self.SoundDatabase.corpora[corpus_index].fileDescriptors['norm'][i]

                p += 1

        self.atoms = rfn.stack_arrays((self.atoms, da)).data
        self.count()

    def mp(self, signal, cmax, srr_thresh):

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))

        #M = np.zeros(cmax, dtype=dtype)
        M = pydbm.book.SoundgrainBook(self.sampleRate, self.SoundDatabase, cmax)
        
        #place to put model
        out = np.zeros(len(signal), dtype=float)

        #place to hold analysis values
        max_mag = np.zeros(len(self.atoms))
        max_ind = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))
        max_scale = max(self.atoms['duration'])

        #breaking conditions
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:
                
                atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].soundfiles[self.atoms['file_index'][cnt]])[0]  
                #apply scalar to scalar rather than array for savings
                a = np.inner(atom, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / self.atoms['norm'][cnt]
                max_mag[cnt] = a    
                
            #get and remove maximally correlated atom
            indx = np.argmax(max_mag)

            a = max_mag[indx]
            atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].soundfiles[self.atoms['file_index'][indx]])[0]  / self.atoms['norm'][indx]  

            signal[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] -= atom * a
            out[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] += atom * a
            
            #Store decomposition Values
            M.atoms['type'][c_cnt] = 'soundgrain'
            M.atoms['duration'][c_cnt] = self.atoms['duration'][indx]
            M.atoms['onset'][c_cnt] = self.atoms['onset'][indx]
            M.atoms['corpus_index'][c_cnt] = self.atoms['corpus_index'][indx]
            M.atoms['file_index'][c_cnt] = self.atoms['file_index'][indx]
            M.atoms['norm'][c_cnt] = self.atoms['norm'][indx]
            M.atoms['mag'][c_cnt] = a

            #Measure the change    
            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2) 
            print(a**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)            

            up_ind = np.intersect1d(np.where(self.atoms['onset'] >= self.atoms['onset'][indx] - max_scale)[0], np.where(self.atoms['onset'] < self.atoms['onset'][indx] + max_scale)[0]) 
            max_mag[up_ind] = 0.

            c_cnt += 1

        M.atoms = M.atoms[0:c_cnt]

        return out, signal, M

    def tvmp(self, signal, cmax, srr_thresh, globscalar):

        '''Experimental variant of MP that does not re-scale the samples'''

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))

        #M = np.zeros(cmax, dtype=dtype)
        M = pydbm.book.SoundgrainBook(self.sampleRate, self.SoundDatabase, cmax)
        
        #place to put model
        out = np.zeros(len(signal), dtype=float)

        #place to hold analysis values
        max_mag = np.zeros(len(self.atoms))
        max_ind = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))
        max_scale = max(self.atoms['duration'])

        #breaking conditions
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0
        start = 1.

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:
                
                atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].soundfiles[self.atoms['file_index'][cnt]])[0]  
                a = np.inner(atom * globscalar, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]])
                max_mag[cnt] = a    
                
            #get and remove maximally correlated atom
            indx = np.argmax(max_mag)

            a = max_mag[indx]
            atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].soundfiles[self.atoms['file_index'][indx]])[0]  

            signal[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] -= atom * globscalar
            out[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] += atom * globscalar
            
            #Store decomposition Values
            M.atoms['type'][c_cnt] = 'soundgrain'
            M.atoms['duration'][c_cnt] = self.atoms['duration'][indx]
            M.atoms['onset'][c_cnt] = self.atoms['onset'][indx]
            M.atoms['corpus_index'][c_cnt] = self.atoms['corpus_index'][indx]
            M.atoms['file_index'][c_cnt] = self.atoms['file_index'][indx]
            M.atoms['norm'][c_cnt] = self.atoms['norm'][indx]
            M.atoms['mag'][c_cnt] = a

            #Measure the change    
            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2)
            perc = linalg.norm(signal)**2 / start_norm**2
            if perc > start:
                break
            start = perc
  
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)            

            up_ind = np.intersect1d(np.where(self.atoms['onset'] >= self.atoms['onset'][indx] - max_scale)[0], np.where(self.atoms['onset'] < self.atoms['onset'][indx] + max_scale)[0]) 
            max_mag[up_ind] = 0.

            c_cnt += 1

        M.atoms = M.atoms[0:c_cnt]

        return out, signal, M

    def mpc(self, signal, cmax, srr_thresh, maxsimul, mindistance):

        dtype = self.atoms.dtype.descr
        dtype.append(('mag', float))
        M = pydbm.book.SoundgrainBook(self.sampleRate, self.SoundDatabase, cmax)
        #M.atoms['onset'] = np.inf #set initial onsets to inf so that 0 onset is initially acceptable
        out = np.zeros(len(signal), dtype=float)
        max_mag = np.zeros(len(self.atoms))
        max_ind = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))
        max_scale = max(self.atoms['duration'])
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):
            print(c_cnt)
            for cnt in up_ind:
                atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].soundfiles[self.atoms['file_index'][cnt]])[0]  
                a = np.inner(atom, signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt]]) / self.atoms['norm'][cnt]
                max_mag[cnt] = a    

            #Application of constraints
            if c_cnt == 0:
                indx = np.argmax(max_mag)
                a = max_mag[indx]
                up_ind = np.intersect1d(np.where(self.atoms['onset'] >= self.atoms['onset'][indx] - max_scale)[0],
                                            np.where(self.atoms['onset'] < self.atoms['onset'][indx] + max_scale)[0])
                max_mag[up_ind] = 0.
                
            else:
                mag_sort = np.argsort(max_mag)[::-1]    
                indx = None
                for magi in mag_sort:
                    constraint1 = sum(np.where(M.atoms['onset'][0:c_cnt] == self.atoms['onset'][magi])[0]) <= maxsimul
                    constraint2 = all(abs(M.atoms['onset'][0:c_cnt] - self.atoms['onset'][magi]) >= mindistance) or self.atoms['onset'][magi] in M.atoms['onset'][0:c_cnt]
                    
                    if constraint1 and constraint2:
                        indx = magi
                        a = max_mag[indx]
                        up_ind = np.intersect1d(np.where(self.atoms['onset'] >= self.atoms['onset'][indx] - max_scale)[0],
                                            np.where(self.atoms['onset'] < self.atoms['onset'][indx] + max_scale)[0])
                        max_mag[up_ind] = 0.
                        up_ind = up_ind[np.union1d(np.where(abs(self.atoms['onset'] - self.atoms['onset'][magi]) >= mindistance)[0],
                                                   np.where(self.atoms['onset'] in M.atoms['onset'][0:c_cnt])[0])]
                        break

            if not indx:
                print('No atoms satisfy the given constraints')
                break

            atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].soundfiles[self.atoms['file_index'][indx]])[0]  / self.atoms['norm'][indx]  

            signal[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] -= atom * a
            out[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] += atom * a
            
            #Store decomposition Values
            M.atoms['type'][c_cnt] = 'soundgrain'
            M.atoms['duration'][c_cnt] = self.atoms['duration'][indx]
            M.atoms['onset'][c_cnt] = self.atoms['onset'][indx]
            M.atoms['corpus_index'][c_cnt] = self.atoms['corpus_index'][indx]
            M.atoms['file_index'][c_cnt] = self.atoms['file_index'][indx]
            M.atoms['norm'][c_cnt] = self.atoms['norm'][indx]
            M.atoms['mag'][c_cnt] = a

            #Measure the change    
            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2) 
            print(a**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)
            
            c_cnt += 1

        M.atoms = M.atoms[0:c_cnt]

        return out, signal, M

    #FIX 
    def mp_stereo(self, signal, cmax, srr_thresh):
        '''Matching Pursuit for stereo sound grains'''

        #initialize
        dtype = self.atoms.dtype.descr
        dtype.append(('mag', '2float'))

        #M = np.zeros(cmax, dtype=dtype)
        M = pydbm.book.SoundgrainBook(self.sampleRate, self.SoundDatabase, cmax)
        
        #place to put model
        out = np.zeros(len(signal), dtype=float)

        #place to hold analysis values
        max_mag = np.zeros(len(self.atoms), dtype='2float')
        max_ind = np.zeros(len(self.atoms))
        up_ind = np.arange(len(self.atoms))
        max_scale = max(self.atoms['duration'])

        #breaking conditions
        start_norm = linalg.norm(signal)
        srr = 0.
        c_cnt = 0

        while (c_cnt < cmax) and (srr <= srr_thresh):

            print(c_cnt)

            for cnt in up_ind:
                
                atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][cnt]].soundfiles[self.atoms['file_index'][cnt]])[0]  

                for chan in (0, 1):

                    a = np.inner(atom[0:, chan], signal[self.atoms['onset'][cnt]:self.atoms['onset'][cnt]+self.atoms['duration'][cnt], chan]) / self.atoms['norm'][cnt]
                    max_mag[cnt][chan] = a    
                
            #get and remove maximally correlated atom
            indx = np.argmax(np.sum(max_mag, axis=1))

            a = max_mag[indx]
            atom = self.readAudio(self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].directory + '/' + self.SoundDatabase.corpora[self.atoms['corpus_index'][indx]].soundfiles[self.atoms['file_index'][indx]])[0]  / self.atoms['norm'][indx]  

            signal[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] -= atom * a
            out[self.atoms['onset'][indx] : self.atoms['onset'][indx]+self.atoms['duration'][indx]] += atom * a
            
            #Store decomposition Values
            M.atoms['type'][c_cnt] = 'soundgrain'
            M.atoms['duration'][c_cnt] = self.atoms['duration'][indx]
            M.atoms['onset'][c_cnt] = self.atoms['onset'][indx]
            M.atoms['corpus_index'][c_cnt] = self.atoms['corpus_index'][indx]
            M.atoms['file_index'][c_cnt] = self.atoms['file_index'][indx]
            M.atoms['norm'][c_cnt] = self.atoms['norm'][indx]
            M.atoms['mag'][c_cnt] = a

            #Measure the change    
            srr = 10 * np.log10(linalg.norm(out)**2 / linalg.norm(signal)**2) 
            print(a**2 / start_norm**2)
            print(linalg.norm(signal)**2 / start_norm**2)
            print(srr)            

            up_ind = np.intersect1d(np.where(self.atoms['onset'] >= self.atoms['onset'][indx] - max_scale)[0], np.where(self.atoms['onset'] < self.atoms['onset'][indx] + max_scale)[0]) 
            max_mag[up_ind] = 0.

            c_cnt += 1

        M.atoms = M.atoms[0:c_cnt]

        return out, signal, M

        

