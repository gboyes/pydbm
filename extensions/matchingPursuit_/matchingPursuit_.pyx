import cython

import numpy as np
import pydbm.book

cimport numpy as np
import atom_

#C external functions
cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double cos(double)
    double sin(double)
    double atan2(double, double)

cdef double pi = 3.1415926535897931

#Gaussian windows
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double gauss(int i, int N):
    return exp(-( (i - N/2. )**2) / (2. * ( (0.125 * N)**2)))

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double realSinusoid(int i, float omega, float phi):
    return cos(2 * pi * omega * i + phi)

'''
#MPTK's
cdef double gauss(int i, int N):
    return exp( -(i -(N/2. - 1) / 2.) * (i -(N/2. - 1) / 2.) * (1./(2 * 0.02 * (N/2. +1)**2))) 

cpdef double gauss_(int i, int N):
    return exp( -((i -(N/2. - 1)) / 2.) * ((i -(N/2. - 1)) / 2.) * (1./(2 * 0.02 * (N/2. +1)**2))) 

#Gabor's
cdef double gauss(int i, int N):
    cdef double alpha = 0.005
    return exp( -(alpha**2) * (i - N/2.)**2)
'''

#two point norm to measure magnitude of coefficient
@cython.profile(False)
@cython.boundscheck(False)
cdef double norm(double c, double s):
    return sqrt(c**2 + s**2)

@cython.profile(False)
@cython.boundscheck(False)
cdef double vnorm(np.ndarray[np.float_t, ndim=1, mode='c'] x):

    cdef float i
    cdef double s = 0.

    for i in x:
        s = s + i**2

    s = sqrt(s)

    return s

@cython.profile(False)
@cython.boundscheck(False)
cdef int argmax(np.ndarray[np.float_t, ndim=1, mode="c"] ar, int length):
    cdef double m = 0.
    cdef int i
    for k in range(length):
        if ar[k] > m:
            m = ar[k]
            i = k
    return i  

'''
@cython.nonecheck(False)
@cython.boundscheck(False)
def matchingPursuit_(np.ndarray[np.float_t, ndim=1, mode="c"] signal, dictionary, np.ndarray[np.int32_t, ndim=1, mode="c"] N, int cmax, double pmdc, double last_measure, double start_norm):
    
    #initialization
    cdef double delta_measure, measure, m1, mc0, mc1, mag, c_omega, c_mag, c_phase, cn1, cn2, a_point, delta_thresh
    
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] locs = np.array(dictionary['onset'])
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] scales = np.array(dictionary['scale'])
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] omegas = np.array(dictionary['omega'])
    
    cdef int c_cnt = 0, L = len(signal), L_N = len(N), L_D = len(locs), max_scale = max(dictionary['scale']), cnt, n, cm, start, end, indx, pindx, k, c_scale, c_loc, cs

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] atom1 = np.zeros(max_scale, dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] atom2 = np.zeros(max_scale, dtype=np.float)

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(L)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] max_ind = np.zeros(L_N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] up_ind = np.arange(L_N, dtype=np.int32)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] max_mag = np.zeros(L_N, dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] max_phase = np.zeros(L_N, dtype=np.float)
    
    #to store the molecule contents
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] M_scales = np.zeros(cmax, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] M_locs = np.zeros(cmax, dtype=np.int32)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] M_omegas = np.zeros(cmax, dtype=np.float)    
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] M_mags = np.zeros(cmax, dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] M_phases = np.zeros(cmax, dtype=np.float)

    delta_thresh = last_measure * pmdc
    delta_measure = last_measure

    while (c_cnt < cmax) and (delta_measure > delta_thresh):

        print(c_cnt)
       
        for cnt in up_ind:
            start = N[cnt]

            for k in range(L_D):
                if locs[k]  == start:
                    c_scale = scales[k]
                    c_omega = omegas[k]
                
                    cn1 = 0.
                    cn2 = 0.
                
                    #build atoms and calculate norms
                    for cs in range(c_scale):

                        a1 = gauss(cs, c_scale) * cos(2 * pi * c_omega * cs)
                        #a1 = gauss(cs, c_scale) * realSinusoid(cs, c_omega, 0.)
                        cn1 += a1*a1
                        
                        a2 = gauss(cs, c_scale) * -sin(2 * pi * c_omega * cs)
                        #a2 = gauss(cs, c_scale) * realSinusoid(cs, c_omega, pi/2.)
                        cn2 += a2*a2
                        
                        atom1[cs] = a1
                        atom2[cs] = a2
                
                    a1 = sqrt(cn1)
                    a2 = sqrt(cn2)

                    for cs in range(c_scale):
                        atom1[cs] /= a1
                        atom2[cs] /= a2

                    cn1 = 0.
                    cn2 = 0.
                
                    #inner product
                    for cs in range(c_scale):
                        cn1 += atom1[cs] * signal[start+cs]
                        cn2 += atom2[cs] * signal[start+cs]
                        
                    mag = norm(cn1, cn2)

                    #store max values (coefficient and corresponding index) for each time interval, i.e. best atom starting at a given point     
                    if mag > max_mag[cnt]:
                        max_mag[cnt] = mag
                        max_phase[cnt] = atan2(cn2, cn1) 
                        max_ind[cnt] = k

        #get and remove maximally correlated atom
        indx = argmax(max_mag, L_N)
        pindx = max_ind[indx]

        c_scale = scales[pindx]
        c_loc = locs[pindx] 
        c_omega = omegas[pindx]
        c_phase = max_phase[indx]
        c_mag = max_mag[indx]

        cn1 = 0.

        #build atom sample by sample
        for cs in range(c_scale):

            a1 = gauss(cs, c_scale) * cos(2 * pi * c_omega * cs + c_phase)
            #a1 = gauss(cs, c_scale) * realSinusoid(cs, c_omega, c_phase)
            cn1 += a1*a1
            atom1[cs] = a1
                
        a1 = sqrt(cn1)
        for cs in range(c_scale):
            atom1[cs] /= a1

        #subtract from signal and add to synthesis sample by sample
        for cs in range(c_scale):
            signal[c_loc+cs] -= atom1[cs] * c_mag
            out[c_loc+cs] += atom1[cs] * c_mag

        #add to molecule synth Book
        M_scales[c_cnt] = c_scale
        M_locs[c_cnt] = c_loc
        M_omegas[c_cnt] = c_omega
        M_mags[c_cnt] =  c_mag
        M_phases[c_cnt] = c_phase
                
        #Measure the change    
        m1 = 0. 

        for cm in range(L):
            m1 += signal[cm]**2
        
        #the measurement value is the squared norm.
        measure = m1/start_norm**2
        delta_measure = last_measure - measure
        last_measure = measure

        print(measure)
        print(delta_measure)
	
        #find which indices to update
        UL = []
        for cs in range(L_N):
            n = N[cs]
            if (n >= c_loc - max_scale) and (n < c_loc + max_scale):
                UL.append(cs)
                max_mag[cs] = 0.
        
        up_ind = np.array(UL)

        c_cnt += 1

    M = {'scale': M_scales[0:c_cnt], 'onset': M_locs[0:c_cnt], 'omega': M_omegas[0:c_cnt], 'mag': M_mags[0:c_cnt], 'phase' : M_phases[0:c_cnt]}
    
    return out, signal, M
'''
'''
@cython.profile(False)
@cython.boundscheck(False)
def matchingPursuit_(np.ndarray[np.float_t, ndim=1, mode='c'] signal, dictionary, int cmax, double srr_thresh):

    #initialize
    dtype = dictionary.dtype.descr
    dtype.append(('index', int))
    dtype.append(('mag', float))
    dtype.append(('phase', float))
    dtype.append(('norm', float))
        
    B = pydbm.book.Book(cmax, dtype, self.sampleRate)

    #place to put model
    out = np.zeros(len(signal), dtype=float)

    #place to hold analysis values
    max_mag = np.zeros(len(dictionary))
    max_phase = np.zeros(len(dictionary))
    up_ind = np.arange(len(dictionary))

    #initialize breaking condition
    start_norm = vnorm(signal)
    srr = 0.
    c_cnt = 0

    while (c_cnt < cmax) and (srr <= srr_thresh):

        print(c_cnt)

        for cnt in up_ind:

            #protip: the first number is phase ;)
            atom1 = self.atomGenTable[dictionary['type'][cnt]].gen(0., *[dictionary[cnt][arg] for arg in self.atomGenTable[dictionary['type'][cnt]].genargs])
            atom2 = self.atomGenTable[dictionary['type'][cnt]].gen(np.pi/2, *[dictionary[cnt][arg] for arg in self.atomGenTable[dictionary['type'][cnt]].genargs])
                    
            a1 = np.inner(atom1, signal[dictionary['onset'][cnt]:dictionary['onset'][cnt]+dictionary['scale'][cnt]]) / vnorm(atom1)
            a2 = np.inner(atom2, signal[dictionary['onset'][cnt]:dictionary['onset'][cnt]+dictionary['scale'][cnt]]) / vnorm(atom2)

            max_mag[cnt] = np.sqrt(a1**2 + a2**2)
            max_phase[cnt] = np.arctan2(a2, a1)

        #get the parameters of the maximally correlated atom
        indx = np.argmax(max_mag)

        #get and remove maximally correlated atom
        indx = np.argmax(max_mag)

        mag = max_mag[indx]
        phase = max_phase[indx]
        atom = self.atomGenTable[dictionary['type'][indx]].gen(phase, *[dictionary[indx][arg] for arg in self.atomGenTable[dictionary['type'][indx]].genargs])

        norman = vnorm(atom)
        atom *= 1./norman

        signal[dictionary['onset'][indx] : dictionary['onset'][indx]+dictionary['scale'][indx]] =  signal[dictionary['onset'][indx] : dictionary['onset'][indx]+dictionary['scale'][indx]] - (atom * mag)
        out[dictionary['onset'][indx] : dictionary['onset'][indx]+dictionary['scale'][indx]] = out[dictionary['onset'][indx] : dictionary['onset'][indx]+dictionary['scale'][indx]] + (atom * mag)
            
        #Store decomposition Values
        B.atoms['type'][c_cnt] = dictionary['type'][indx]
        B.atoms['onset'][c_cnt] = dictionary['onset'][indx]
        B.atoms['mag'][c_cnt] = mag
        B.atoms['phase'][c_cnt] = phase
        B.atoms['norm'][c_cnt] = norman 

        #set the free args
        fixedarg = ['type', 'onset', 'mag', 'phase', 'norm']
        for param in [P[0] for P in self.atomGenTable[dictionary[indx]['type']].dictionaryType if P[0] not in fixedarg]: 
            B.atoms[param][c_cnt] = dictionary[indx][param]

        srr = 10 * np.log10( vnorm(out)**2 / vnorm(signal)**2 ) 
        print(mag**2 / start_norm**2)
        print(vnorm(signal)**2 / start_norm**2)
        print(srr)
 
        #indices to update
        up_ind = np.union1d(np.intersect1d(np.where(dictionary['onset'] + dictionary['scale'] < dictionary[indx]['onset'] + dictionary[indx]['scale'])[0], 
                            np.where(dictionary['onset'] + dictionary['scale'] >= dictionary[indx]['onset'])[0]),
                            np.intersect1d(np.where(dictionary['onset'] < dictionary[indx]['onset'] + dictionary[indx]['scale'])[0], 
                            np.where(dictionary['onset']  >= dictionary[indx]['onset'])[0]))
            
        max_mag[up_ind] = 0.
        c_cnt += 1

    B.atoms = B.atoms[0:c_cnt]
    B.model = np.array(out)
    B.residual = np.array(signal)
    B.resPercent =  (vnorm(B.residual)**2 / start_norm**2) * 100
    B.srr = srr

    return out, signal, B
'''