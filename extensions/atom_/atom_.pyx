cimport cython
import numpy as np
cimport numpy as np

#C external functions
cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double cos(double)
    double sin(double)
    double atan2(double, double)
    double log(double)

cdef double pi = 3.1415926535897931


#Gaussian for C
############################################################
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double gauss(int i, int N):
    return exp(-( (i - N/2. )**2) / (2. * ( (0.125 * N)**2)))

#Hann for C#
############
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double hann(int i, int N):
    return 0.5 * (1 - cos( 2*pi*i / (N-1)))

#Blackman for C#
################
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double blackman(int i, int N):
    return 0.42 - 0.5 * cos(2*pi*i/(N-1)) + 0.08 * cos(4*pi*i/(N-1)) 

#Real Sinusoid for C
###########################################################################
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double realSinusoid(int i, float omega, float chirp, float phi):
    return cos(2 * pi * (omega + 0.5 * chirp * i) * i + phi)

#Real Vibrating Sinusoid for C
###########################################################################
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double realSinusoidFM(int i, float omega, float chirp, float phi, float omega_m, float phi_m, depth):
    return cos(2 * pi * (omega + 0.5 * chirp * i) * i + (depth/omega_m * sin(2 * pi * omega_m * i + phi_m)) + phi)

#Real Sinusoid for python
######################################################################################
@cython.profile(False)
@cython.boundscheck(False) 
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] realSinusoid_(int N, float omega, float chirp, float phi):
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.cos(2 * pi * (omega + chirp * 0.5 * np.arange(N)) * np.arange(N) + phi)
    return out

#Gabor for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gabor_(float phi, int N, float omega, float chirp):
    '''A gabor atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = gauss(i, N) * realSinusoid(i, omega, chirp/N, phi) 

    return out

#GaborFM for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gaborFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):
    '''A gabor atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = gauss(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth) 

    return out

#Hann for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hann_(float phi, int N, float omega, float chirp):
    '''A hann atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = hann(i, N) * realSinusoid(i, omega, chirp/N, phi) 

    return out

#HannFM for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hannFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):
    '''A hann atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = hann(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth) 

    return out

#Blackman for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackman_(float phi, int N, float omega, float chirp):
    '''A blackman atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = blackman(i, N) * realSinusoid(i, omega, chirp/N, phi) 

    return out

#BlackmanFM for python
#####################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackmanFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):
    '''A blackman FM atom where
       phi := initial phase
       N := scale, i.e. length
       omega := normalized frequency''' 
       
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = blackman(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth) 

    return out


#Gamma for python
##################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gamma_(float phi, int N, float omega, float chirp, float order, float bandwidth):

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoid(i, omega, chirp/N, phi) 

    return out

#GammaFM for python
##################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gammaFM_(float phi, int N, float omega, float chirp, float order, float bandwidth, float omega_m=0., float phi_m=0., float depth=0.):

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth) 

    return out


#Damped for python
##################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] damped_(float phi, int N, float omega, float chirp, float damp):

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = exp(-damp * i) * realSinusoid(i, omega, chirp/N, phi) 

    return out

#DampedFM for python
##################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] dampedFM_(float phi, int N, float omega, float chirp, float damp, float omega_m=0., float phi_m=0., float depth=0.):

    cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
    cdef int i
    
    for i in range(N):
        out[i] = exp(-damp * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)

    return out

    

#FOF 
#################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fof_(float phi, int N, float omega, float chirp, int rise_n, int decay_n):
    
    cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
    cdef int t
    cdef float op = log(decay_n) / decay_n
    cdef float factor = pi/rise_n
    cdef float p, a
 
    for t in range(rise_n):
        out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
    
    p = out[rise_n-1]
    for t in range(rise_n, N):
        out[t-1] = exp(-op*t)

    a = max(abs(out[rise_n-1:N-1]))
    for t in range(rise_n-1, N-1):
        out[t] = out[t]/a  * p

    for i in range(N):
        out[i] = out[i] * realSinusoid(i, omega, chirp/N, phi)
    
    return out

#FOFFM 
#################################################################################################################
@cython.profile(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fofFM_(float phi, int N, float omega, float chirp, int rise_n, int decay_n, float omega_m=0., float phi_m=0., float depth=0.):
    
    cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
    cdef int t
    cdef float op = log(decay_n) / decay_n
    cdef float factor = pi/rise_n
    cdef float p, a
 
    for t in range(rise_n):
        out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
    
    p = out[rise_n-1]
    for t in range(rise_n, N):
        out[t-1] = exp(-op*t)

    a = max(abs(out[rise_n-1:N-1]))
    for t in range(rise_n-1, N-1):
        out[t] = out[t]/a  * p

    for i in range(N):
        out[i] = out[i] * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
    
    return out
    
    

    
    
    

 
