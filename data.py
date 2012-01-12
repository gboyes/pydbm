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

import os
import re

import pysdif
import numpy as np
import scipy.linalg as linalg
import xml.etree.ElementTree as etree 

import pydbm.meta
import pydbm.utils


#Class for to obtain and store information related to a particular instrument subspace#
#######################################################################################

class InstrumentSubspace(pydbm.meta.IO, pydbm.meta.Spectral, pydbm.utils.Utils):
    '''Class for data related a particular instrument subspace'''

    def __init__(self, fs, instrument):
        IO.__init__(self)
        self.sampleRate = fs
        self.instrument = instrument

    def getSpectralPeaks(self, trainingCorpora, W, hop, Nfft, minPeaks, maxPeaks, dBthresh, window, interp=True, **winargs):
        '''Get the partial data for an instrument from a list of labeled corpora, where
           trainingCorpora := a list of directories to use to build the instrument model containing labeled exemplars
           W := analysis window size
           hop := analysis hop size
           Nfft := number of spectral bins
           maxPeaks := maximum number of spectral peaks
           dBthresh := threshold to stop peak search
           window := window function to be used, these are taken from the atom generator objects
           winargs := any window specific arguments'''

        dtype = [('duration', int), ('omega', float), ('amplitude', float), ('phase', float)]
        self.envelopes = {}# a midicents (pitch class) to list of envelope structured arrays look up table

        for corpus in trainingCorpora:
            
            for fn in os.listdir(corpus):

                #ignore hidden files or incompatible files
                if (fn[0] == '.') or (os.path.splitext(fn)[1] not in self.informats.keys()):
                    continue  
                
                x, fs = self.readAudio(corpus + '/' + fn)
                z = os.path.basename(fn)
          
                m = re.search('[0-9]+', z).group()                                                                                            
                midi = eval(m)

                #make a list to hold Pitch objects, if it's not already there
                if not midi in self.envelopes.keys():
                    self.envelopes[midi] = []

                freq = self.midi2hz(midi / 100.) / float(fs)
                minSpace = int(np.round((freq / 2.) / (1./Nfft)))
                tol = 2*(1./Nfft)

                L = len(x)
                xp = np.concatenate((np.zeros(W), x, W - np.zeros(np.mod(L, hop))))
                num = len(xp)/hop

                pin = 0
                pend = len(xp) - 2*W

                while pin < pend:

                    X = fftpack.fft(xp[pin:pin+W] * self.atomGenTable[window].window(W, **winargs), Nfft) 
                    R = abs(X)
                    P = np.angle(X)

                    L, V = self.pickPeaks(R, maxPeaks, dBthresh, minSpace, W, window, **winargs) 

                    if interp:
                        L, V, P = self.interpolateValues(R, P, Nfft, L, V, W, window, **winargs)

                    f = L/float(Nfft)

                    #remove anything below fundamental                                                                                                                            
                    ind = np.where(f >= freq-tol)[0]
                    f = f[ind]

                    if len(f) <= minPeaks:
                        pin += hop
                        continue

                    p = P[ind]
                    a = 2 * V[ind] / W * (1./self.atomGenTable[window].coherentGain(W, **winargs))#linear amplitude values from magnitude

                    i = np.argsort(f)

                    env = np.zeros(len(i), dtype=dtype)
                    env['omega'] = f[i]
                    env['amplitude'] = a[i]
                    env['phase'] = p[i]
                    env['duration'] = W

                    self.envelopes[midi].append(env)
                                                                                 
                    pin += hop

    def round(self, dec=3):
        '''Round the peak frequencies to a given number of decimals and find average amplitude and phase values for each unique frequency
           this is a relatively naive way to limit the number of elements in a subspace, note that normalization will not be valid after rounding''' 

        for k in self.envelopes.keys():

            for array in self.envelopes[k]:
                array['omega'] = np.round(array['omega'], dec)

            u = reduce(np.union1d, [a['omega'] for a in self.envelopes[k]])
            env = np.zeros(len(u), dtype=[('duration', int), ('omega', float), ('amplitude', float), ('phase', float)])
            env['duration'] = self.envelopes[k][0]['duration'][0]

            for omi, om in enumerate(u):
                b = [np.where(a['omega'] == om)[0] for a in self.envelopes[k]]
                env['amplitude'][omi] = np.mean(reduce(np.union1d, [self.envelopes[k][i]['amplitude'][b[i]] for i in xrange(len(b))]))
                env['phase'][omi] = np.mean(reduce(np.union1d, [self.envelopes[k][i]['phase'][b[i]] for i in xrange(len(b))]))
                env['omega'][omi] = om

            self.envelopes[k] = [env]

    def normalize(self):
        '''Normalize the partial amplitudes such that sum(a**2) = 1.'''
 
        for key in self.envelopes.keys():

            for a in self.envelopes[key]:

                a['amplitude'] *= 1./np.sqrt(np.sum(a['amplitude']**2))

    #FIX
    '''
    def cluster(self, total):
        
        for key in self.partialAmplitudes.keys():

            nums = np.unique([len(a) for a in self.partialAmplitudes[key]])
            tmp_a = []
            tmp_p = []
            tmp_f = []

            for n in nums:

                inds = np.where(np.array([len(a) for a in self.partialAmplitudes[key]]) == n)[0]
                M = np.zeros((len(inds), n))
                P = M.copy()
                F = P.copy()

                for j, i in enumerate(inds):

                    M[j, 0:] = self.partialAmplitudes[key][i]
                    P[j, 0:] = self.partialPhases[key][i]
                    F[j, 0:] = self.partialFrequencies[key][i]
 
                centroids_a, dist_a = vq.kmeans(M, total)
                centroids_p, dist_p = vq.kmeans(P, total)
                #centroids_f, dist_f = vq.kmeans(F, (np.arange(1, n+1) * self.midi2hz(key / 100.) / self.sampleRate).reshape(1, n))
                centroids_f, dist_f = vq.kmeans(F, total)                

                maxK = min([len(centroids_a), len(centroids_p), len(centroids_f)]) 
                tmp_a.append([w for w in centroids_a[0:maxK]])
                tmp_p.append([w for w in centroids_p[0:maxK]])
                tmp_f.append([w for w in centroids_f[0:maxK]])
                #tmp_f.append([self.partialFrequencies[key][inds[w]] for w in xrange(total)])

            self.partialAmplitudes[key] = [item for sublist in tmp_a for item in sublist]
            self.partialPhases[key] = [item for sublist in tmp_p for item in sublist]
            self.partialFrequencies[key] = [item for sublist in tmp_f for item in sublist]
    '''


#Sound database and Corpus objects        
######################################################################################################
class SoundDatabase(pydbm.meta.IO):

    def __init__(self, list_of_corpora):
        pydbm.meta.IO.__init__(self)
        self.corpora = [Corpus(k) for k in list_of_corpora]
        
    def num(self):
        sum([C.num() for C in self.corpora]) 

class Corpus(pydbm.meta.IO):

    def __init__(self, directory):
        pydbm.meta.IO.__init__(self)
        self.directory = directory
        self.getSoundfiles()

    def num(self):
        return len(self.soundfiles)

    def getSoundfiles(self):
        
        self.soundfiles = []
        
        #get only the soundfiles, avoid anything else in the dir
        #d = os.listdir(self.directory)

        d = [q for q in os.listdir(self.directory) if q[0] != '.']
        self.fileDescriptors = np.zeros(len(d), dtype=[('file_index', int), ('length', int), ('sampleRate', int), ('norm', float)])
        
        i = 0
        k = 0
        for ind, val in enumerate(d):

            if not any([os.path.splitext(val)[1].lower() == p for p in ['.aif', '.wav', '.aiff', '.au']]):
                continue

            x = self.readAudio(self.directory +'/'+ val)
            self.fileDescriptors['file_index'][i] = k
            self.fileDescriptors['length'][i] = len(x[0])
            self.fileDescriptors['sampleRate'][i] = x[1]
            self.fileDescriptors['norm'][i] = linalg.norm(x[0])
            i += 1
            k += 1

            self.soundfiles.append(val)
        
        self.fileDescriptors = self.fileDescriptors[0:i]

    
    def getSignalDescriptors(self):
        '''Fill a database with signal descriptors for a sound corpus'''
        #idea: user should be able to give a list of built-in descriptor types

        self.signalDescriptors = {'norm' : np.zeros(sum(self.cardinality), dtype=float)}

        i = 0
        for indx, c in enumerate(self.soundfiles):

            for f in c:
                x = self.readAudio(self.corpora[indx] +'/'+ f)

                #get some audio descriptors here!
                self.signalDescriptors['norm'][i] = linalg.norm(x[0])

                i += 1

    def getLabelDescriptors(self):
        '''Fill database of label descriptors about a sound corpus
           Note that these correspond a specific convention, e.g. foo_123-456.wav
           where the first set of consecutive numbers is midicents and the second is midi velocity'''

        #this should be made more flexible
        self.labelDescriptors = {'midicents' : np.zeros(sum(self.cardinality), dtype=int), 'velocity' : np.zeros(sum(self.cardinality), dtype=int)}

        i = 0
        for c in self.soundfiles:
 
            for f in c:

                z = f.partition('-')
                m = re.search('[0-9]+', z[0]) #the part before the partition char
                v = re.search('[0-9]+', z[2]) #the part after the partition char
                self.labelDescriptors['midicents'][i] = eval(m.group())
                self.labelDescriptors['velocity'][i] = eval(v.group())

                i += 1
      

#Partial helper classes#
########################

class PartialModel(pydbm.meta.IO):
    '''A class which contains a set of partials and methods to treat them from a SDIF file'''

    def __init__(self, sdifpath):
        pydbm.meta.IO.__init__(self)
        self.partialModel = self.sdif2array(sdifpath, ['1TRC'])['1TRC']
        self.getPartials()

    def getPartials(self):
        self.partials = [Partial(i, self.partialModel[np.where(self.partialModel['index'] == i)[0]]) for i in np.unique(self.partialModel['index'])]

    def partialSort(self, order='frequency'):
        '''Sort partials according to some feature'''
         
        inds = {'amplitude' : np.argsort([sum(abs(p.array['amplitude'])) for p in self.partials])[::-1],
                'frequency' : np.argsort([np.mean(p.array['frequency']) for p in self.partials]),
                'duration' : np.argsort([np.max(p.array['time']) - np.min(p.array['time']) for p in self.partials])[::-1],
                'onset' : np.argsort([np.min(p.array['time']) for p in self.partials])}

        self.partials = [self.partials[i] for i in inds[order]]

    def partialDistance(self, alpha=1.0, beta=1.0):
        '''Calculate the proximity of each partial to each partial
        alpha := the weighting coefficient applied to time
        beta := the weighting coefficient applied to frequency'''

        #for the purpose of normalization
        max_t = max([max(p.array['time']) for p in self.partials])
        max_f = max([max(p.array['frequency']) for p in self.partials])

        whole = []

        for k in xrange(len(self.partials)):

            a = self.partials[k]
            whole.append([])

            for i in xrange(len(self.partials)):
                s = 0.
                b = self.partials[i]

                for n in xrange(min( [len(a.array['time']), len(b.array['time'])] )):
                    s += abs( alpha * (a.array['time'][n]/max_t - b.array['time'][n]/max_t)) + abs( beta * (a.array['frequency'][n]/max_f - b.array['frequency'][n]/max_f))

                whole[k].append(s)

        so = [np.argsort(L) for L in whole]

        return so

class Partial(pydbm.meta.IO):
    '''A Partial object'''

    def __init__(self, index, array): 
        self.index = index
        self.array = array


class PolygonGroup(pydbm.meta.IO):
    '''A container for a set of Polygon instances'''
    
    def __init__(self, sdif_in):
        self.polygons = []
        self.readSdif(sdif_in)

    def readSdif(self, sdif_in):
        
        sdif_file = pysdif.SdifFile(sdif_in)
        
        for frame in sdif_file:

            P = Polygon()
            P.readFrame(frame)
            self.polygons.append(P)

class Polygon(object):
    '''A class for time-frequency regions via SDIF'''
    
    def __init__(self):

        self.matrix_types = ['clss', 'dura', 'freq', 'pnts']
        
    def readFrame(self, sdif_frame):
        '''Read an SDIF 1ASO frame and format its data, sets many useful attributes'''

        self.time = sdif_frame.time
        self.getParams(sdif_frame)
        self.polygonClass = ''.join(chr(i) for i in self.params['clss'][0])
        
        #if it's a polygon (with pnts)
        if 'pnts' in self.params.keys():
           self.points = np.zeros(len(self.params['pnts']), dtype=[('time', float), ('frequency', float)])
           for i, a in enumerate(self.params['pnts']):
               self.points[i]['time'] = a[0]
               self.points[i]['frequency'] = a[1]

        #if it's a rectangle
        else:
            self.points = np.zeros(4, dtype=[('time', float), ('frequency', float)])
            c = 0
            for t in [self.time, self.time + self.params['dura'][0]]:
                for f in self.params['freq'][0]:
                    self.points[c]['time'] = t 
                    self.points[c]['frequency'] = f
                    c += 1

    def getParams(self, sdif_frame):
        '''Format params of a sdif_frame'''

        self.params = {} 
        for matrix in sdif_frame:

            for ind, ts in enumerate(self.matrix_types):

                if matrix.signature == ts:
                    self.params[ts] = matrix.get_data().copy()

    def pip(self, x, y):
        '''Test whether a point (x, y) is inside a polygon'''

        n = len(self.tfPoints)
        inside = False

        p1x,p1y = self.tfPoints[0]
        for i in range(n+1):
            p2x,p2y = self.tfPoints[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def getPolyHull(self, fs, hop, fftsize):
        '''Get the coordinates of points that fall within the polygon for a hypothetical TF grid
           fs := sample rate
           hop := hop size
           fftsize := fft size'''

        #snap polygon points to a hypothetical TF grid, i.e. (sec, Hz) => (hop, bin)
        self.tfPoints = np.zeros(len(self.points), dtype=[('hop', int), ('bin', int)])

        u = 1./fftsize
        self.tfPoints['hop'] = np.floor(self.points['time'] * fs / hop)
        self.tfPoints['bin'] = np.round(self.points['frequency'] / fs / u)

        min_t = min(self.tfPoints['hop'])
        min_f = min(self.tfPoints['bin'])
        max_t = max(self.tfPoints['hop'])
        max_f = max(self.tfPoints['bin'])

        self.polyHull = np.zeros((max_t+1 - min_t) * (max_f+1 - min_f), dtype=[('hop', int), ('bin', int)])
        
        #if it's a rectangle, things are very easy...
        if self.polygonClass == 'Rflt' :
            i = 0
            for x in np.arange(min_t, max_t+1):
                for y in np.arange(min_f, max_f+1):
                    self.polyHull[i]['hop'] = x
                    self.polyHull[i]['bin'] = y
                    i+=1 

        #otherwise, a 'free' polygon
        else:
            i = 0
            for x in np.arange(min_t, max_t+1):
                for y in np.arange(min_f, max_f+1):
                    if self.pip(x, y):
                        self.polyHull[i]['hop'] = x
                        self.polyHull[i]['bin'] = y
                        i+=1   
            self.polyHull = np.unique(np.union1d(self.polyHull[0:i], self.tfPoints))



#the 'symbolic' (i.e. score-based') classes#
############################################

class Score(object):
    '''An object for containing ordered symbolic data from musicXML'''

    def __init__(self, musicXMLpath):
        self.XMLtree = etree.parse(musicXMLpath)
        self.midiLookup = {'C-1': 0, 'C#-1': 1, 'D-1': 2, 'D#-1': 3, 'E-1': 4, 'F-1': 5, 'F#-1': 6, 'G-1': 7, 'G#-1': 8, 'A-1' : 9, 'A#-1': 10, 'B-1': 11,
               'C0': 12, 'C#0': 13, 'D0': 14, 'D#0': 15, 'E0': 16, 'F0': 17, 'F#0': 18, 'G0': 19, 'G#0': 20, 'A0' : 21, 'A#0': 22, 'B0': 23,
               'C1': 24, 'C#1': 25, 'D1': 26, 'D#1': 27, 'E1': 28, 'F1': 29, 'F#1': 30, 'G1': 31, 'G#1': 32, 'A1' : 33, 'A#1': 34, 'B1': 35,
               'C2': 36, 'C#2': 37, 'D2': 38, 'D#2': 39, 'E2': 40, 'F2': 41, 'F#2': 42, 'G2': 43, 'G#2': 44, 'A2' : 45, 'A#2': 46, 'B2': 47,
               'C3': 48, 'C#3': 49, 'D3': 50, 'D#3': 51, 'E3': 52, 'F3': 53, 'F#3': 54, 'G3': 55, 'G#3': 56, 'A3' : 57, 'A#3': 58, 'B3': 59,
               'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63, 'E4': 64, 'F4': 65, 'F#4': 66, 'G4': 67, 'G#4': 68, 'A4' : 69, 'A#4': 70, 'B4': 71,
               'C5': 72, 'C#5': 73, 'D5': 74, 'D#5': 75, 'E5': 76, 'F5': 77, 'F#5': 78, 'G5': 79, 'G#5': 80, 'A5' : 81, 'A#5': 82, 'B5': 83,
               'C6': 84, 'C#6': 85, 'D6': 86, 'D#6': 87, 'E6': 88, 'F6': 89, 'F#6': 90, 'G6': 91, 'G#6': 92, 'A6' : 93, 'A#6': 94, 'B6': 95,
               'C7': 96, 'C#7': 97, 'D7': 98, 'D#7': 99, 'E7': 100, 'F7': 101, 'F#7': 102, 'G7': 103, 'G#7': 104, 'A7' : 105, 'A#7': 106, 'B7': 107,
               'C8': 108, 'C#8': 109, 'D8': 110, 'D#8': 111, 'E8': 112, 'F8': 113, 'F#8': 114, 'G8': 115, 'G#8': 116, 'A8' : 117, 'A#8': 118, 'B8': 119,
               'C9': 120, 'C#9': 121, 'D9': 122, 'D#9': 123, 'E9': 124, 'F9': 125, 'F#9': 126, 'G9': 127}
    
    def getParts(self):
        '''Retrieve a list of Part instances from a Score'''
        
        self.parts = []
        root = self.XMLtree.getroot()
        
        for p_ind, part in enumerate(root.findall('part')):

            #a self.Part object, maybe get other attributes?
            self.parts.append(self.Part(p_ind))
    
            for m_ind, measure in enumerate(part.findall('measure')):

                #a Measure object
                self.parts[p_ind].measures.append(self.Measure(m_ind))

                for n_ind, note in enumerate(measure.findall('note')):

                    if note.find('pitch'):

                        #a self.Note object
                        self.parts[p_ind].measures[m_ind].notes.append(self.Note(n_ind))

                        name = note.find('pitch').find('step').text + note.find('pitch').find('octave').text
                        self.parts[p_ind].measures[m_ind].notes[n_ind].name = name
                        self.parts[p_ind].measures[m_ind].notes[n_ind].duration = eval(note.find('duration').text)
                        self.parts[p_ind].measures[m_ind].notes[n_ind].type = note.find('type').text

                        if note.find('pitch').findall('alter'):
                            a = eval(note.find('pitch').find('alter').text)

                        else:
                            a = 0

                        self.parts[p_ind].measures[m_ind].notes[n_ind].alter = a
                        self.parts[p_ind].measures[m_ind].notes[n_ind].midicents = (self.midiLookup[name] + a) * 100

                    elif note.findall('rest'):
     
                        #a self.Rest object
                        self.parts[p_ind].measures[m_ind].notes.append(self.Rest(n_ind))
                        self.parts[p_ind].measures[m_ind].notes[n_ind].duration = eval(note.find('duration').text)
                        self.parts[p_ind].measures[m_ind].notes[n_ind].type = note.find('type').text

    #perhaps add some statistics to these classes 
    class Part(object):
        '''a part from a score'''

        def __init__(self, pnum):
        
            self.partIndex = pnum 
            self.measures = []

    class Measure(object):
        '''a measure from a part'''
    
        def __init__(self, mnum):
            self.measureIndex = mnum
            self.notes = []

    class Note(object):
        '''a note'''
    
        def __init__(self, nnum):
            self.noteIndex = nnum

    class Rest(object):
        '''a rest'''

        def __init__(self, nnum):
            self.noteIndex = nnum

#Class for a particular sound target (stores metadata)
######################################################################################################
class Target(pydbm.meta.IO):
    '''Class for an analysis target'''

    def __init__(self, inpath):
        pydbm.meta.IO.__init__(self)
        self.inpath = inpath
        self.name = os.path.basename(self.inpath)
        self.format = os.path.splitext(self.name)[1]
        x = self.readAudio(self.inpath) 
        self.signal = x[0]
        self.signalLength = len(x[0])
        self.sampleRate = x[1]
