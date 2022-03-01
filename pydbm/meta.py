import os
import re
import copy
import inspect

import numpy as np
import scipy.fftpack as fftpack
import scipy.io.wavfile as wavfile
import scipy.linalg as linalg
import numpy.lib.recfunctions as rfn

from . import atom
from . import utils

# Type management#
#################


class Types(object):
    """SDIF and atom data types are defined here"""

    def __init__(self):

        # automatically get relevant atom generating objects from the atom module and store the instatiated object in a lookup table
        # important to note that they are only instantiated prior to (and not during) the decomposition
        self.atomGenTable = dict(
            [
                (a.type, a)
                for a in [
                    a_[1]()
                    for a_ in inspect.getmembers(atom, inspect.isclass)
                    if "type" in a_[1]().__dict__.keys()
                ]
            ]
        )
        self.atomTypes = self.atomGenTable.keys()
        self.atomGenObjects = self.atomGenTable.values()

        # these are `input' types obtained by reading SDIFs
        self.sdifTypes = {
            "1TRC": [
                ("time", float),
                ("index", int),
                ("frequency", float),
                ("amplitude", float),
                ("phase", float),
            ],
            "XTRD": [("time", float), ("val1", float), ("val2", float)],
            "1CTR": [
                ("time", float),
                ("mxsa", float),
                ("mntd", float),
                ("mxps", float),
                ("mxvs", float),
            ],
            "mxsa": [("time", float), ("max_simultaneous_atoms", float)],
            "mntd": [("time", float), ("min_time_distance", float)],
            "mxps": [("time", float), ("max_pitch_slope", float)],
            "mxvs": [("time", float), ("max_velocity_slope", float)],
            "XASD": [("time", float), ("val1", float), ("val2", float)],
            "XSGR": [
                ("time", float),
                ("onset", int),
                ("corpus_index", int),
                ("file_index", int),
                ("norm", float),
            ],
            "XSLM": [
                ("time", float),
                ("index", int),
                ("onset", int),
                ("duration", int),
                ("corpus_index", int),
                ("file_index", int),
                ("norm", float),
                ("midicents", int),
                ("velocity", int),
                ("mag", float),
            ],
            "XSGM": [
                ("time", float),
                ("index", int),
                ("onset", int),
                ("duration", int),
                ("corpus_index", int),
                ("file_index", int),
                ("norm", float),
                ("mag", float),
            ],
        }

        # here the atom-specific sdif types are added
        for ob in self.atomGenObjects:
            self.sdifTypes[ob.dictionarySdifType] = ob.dictionaryType
            self.sdifTypes[ob.bookSdifType] = ob.inputType


# base class for groups of atom params#
######################################


class Group(object):
    """Abstract base class for both Dictionary and Book objects (methods relevant to a collection of atom params)"""

    # overload __add__ in order to merge Group
    def __add__(self, D):

        C = copy.deepcopy(self)
        Q = copy.deepcopy(D)

        order = C.order + D.order

        # make sure that the operation preserves internal structure
        # Q['index'][Q.order-1] += 1 + max(C['index'][C.order-1])

        C.atoms = rfn.stack_arrays((self.atoms, D.atoms)).data

        return C

    # TODO: structured indexing, add flattening and popping
    def flatten(self):
        pass

    def pop(self):
        pass

    def num(self):
        return len(self.atoms)

    def alter(self, param, function):
        """Modify a parameter of the dictionary
        param := parameter to modify
        function := how to modify it"""

        C = copy.deepcopy(self)
        C.atoms[param] = function(C.atoms[param])

        return C

    def partition(self, indices):
        """Partition a Group according to a set of indices, possibly given some logical criterion e.g. using np.where, returning new objects (the set according to its indices and the complement
        indices := partition according to these"""

        C = copy.deepcopy(self)
        D = copy.deepcopy(self)
        D.atoms = D.atoms[indices]
        C.atoms = C.atoms[np.setxor1d(np.array(indices), np.arange(len(C.atoms)))]

        return D, C


# Input-Output operations (audio and sdif)#
##########################################


class IO(Types):
    """Class for file management.  Functions for reading/writing audio, sdifs are defined here"""

    def __init__(self):
        Types.__init__(self)
        self.informats = {".wav": wavfile.read}
        self.outformats = {".wav": wavfile.write}

    def readAudio(self, path):
        """General wrapper for audio read functions
        path := path to file"""

        f = os.path.splitext(path)
        a = self.informats[f[1].lower()](path)

        return a[1], a[0]

    def writeAudio(self, x, path, fs, format=".wav"):
        """Wrapper for audio write functions
        x := signal
        path := path to new file
        fs := sample rate
        format := output format"""

        self.outformats[format](path + format, fs, x)

    def sdif2array(self, sdif_in, type_string_list):
        """Make an SDIF file into a python dictionary of ndarrays separated by matrix type
        sdif_in := SDIF file to read
        type_string_list := SDIF matrix types to be extracted (these need to be defined in pydbm.data.Type.sdifTypes)"""

        import pysdif

        sdif_file = pysdif.SdifFile(sdif_in)
        data = [[] for k in type_string_list]

        for frame in sdif_file:
            for matrix in frame:
                for ind, ts in enumerate(type_string_list):
                    if matrix.signature == ts:
                        data[ind].append((frame.time, matrix.get_data().copy()))

        num = [sum([len(p[1]) for p in data[k]]) for k in range(len(data))]
        S = dict(
            zip(
                (type_string_list[k] for k in range(len(data))),
                (
                    np.zeros(num[k], dtype=self.sdifTypes[type_string_list[k]])
                    for k in range(len(data))
                ),
            )
        )

        for indx, d in enumerate(data):
            c = 0
            for i in range(len(d)):
                if len(d[i][1]) == 0:
                    continue
                tim = d[i][0]
                for q in range(len(d[i][1])):
                    r = [k for k in d[i][1][q]]
                    r.insert(0, tim)
                    S[type_string_list[indx]][c] = tuple(r)
                    c += 1

        return S

    def mergeArrays(self, array_sequence):
        """Merge a sequence of arrays into a new array"""

        if len(array_sequence) == 1:
            return array_sequence[0]

        elif len(array_sequence) > 1:

            merge = array_sequence[0]

            for i in array_sequence[1:]:
                merge = rfn.stack_arrays((merge, i))

            return merge.data


# Class for typical spectral tools#
##################################


class Spectral(Types):
    """Typical spectral tools"""

    def __init__(self):
        Types.__init__(self)

    def unwrap2pi(self, phi_in):
        """Force phi_in into range [-pi : pi]"""

        phi_in = phi_in - np.floor(phi_in / 2 / np.pi) * 2 * np.pi
        phi_out = phi_in - (phi_in >= np.pi) * 2 * np.pi

        return phi_out

    def spectralShapeStatistics(self, x, Nfft):
        """Spectral shape statistics (centroid, spread, skewness, kurtosis)
        x := signal vector
        Nfft := fft size"""

        mu = lambda fftdata, order: np.sum(
            np.abs(fftpack.fftfreq(Nfft)) ** order * np.abs(fftdata)
        ) / np.sum(np.abs(fftdata))
        F = fftpack.fft(x * sig.hanning(len(x)), Nfft)

        centroid = mu(F, 1)
        spread = np.sqrt(mu(F, 2) - mu(F, 1) ** 2)
        skewness = (
            2 * mu(F, 1) ** 3 - 3 * mu(F, 1) * mu(F, 2) + mu(F, 3)
        ) / spread**3
        kurtosis = (
            -3 * mu(F, 1) ** 4
            + 6 * mu(F, 1) * mu(F, 2)
            - 4 * mu(F, 1) * mu(F, 3)
            + mu(F, 4)
        ) / spread**4 - 3

        return centroid, spread, skewness, kurtosis

    def pickPeaks(
        self, magSpectrum, nPeaks, dBthresh, minSpace, winSize, windowName, **winargs
    ):
        """Classic spectral peak identification algorithm, where:
        magSpectrum := magnitude spectrum
        nPeaks := maximum number of peaks to find
        dBthresh := ignore found peaks below this threshold
        minSpace := number of bins to avoid in search for subsequent peaks
        winSize := analysis window size
        windowName := string assigningthe window ro be used (is taken from atom generator classes)
        winargs := additional window arguments"""

        # window 'coherent gains', i.e. gain applied to make a sinusoid of amplitude 1. = 0 dB
        wcoef = 1.0 / self.atomGenTable[windowName].coherentGain(winSize, **winargs)

        # add a very small value to avoid divide by zero in log10
        eps = 0.0000000001

        # peak search initialization
        N = len(magSpectrum)
        loc = np.zeros(nPeaks)
        val = np.ones(nPeaks) * -100.0

        rmin = min(magSpectrum) - 1

        # the slope of the spectrum
        difference = np.zeros(len(magSpectrum) + 3)
        difference[0] = rmin
        difference[1 : len(difference) - 2] = magSpectrum
        difference[len(difference) - 1] = rmin
        difference = np.diff(difference)

        N_ = int(np.floor(N / 2))

        ilocPP = np.intersect1d(
            np.where(difference[0:N_] >= 0)[0], np.where(difference[1 : N_ + 1] <= 0)[0]
        )
        ivalPP = magSpectrum[ilocPP]

        if len(ivalPP) == 0:
            return [], []

        # find locations and values
        for k in range(nPeaks):
            point = np.argmax(ivalPP)

            # dB
            if (
                20 * np.log10(2 * (ivalPP[point] + eps) / float(winSize) * wcoef)
                < dBthresh
            ):
                break

            val[k] = ivalPP[point]
            loc[k] = ilocPP[point]

            ind = np.where(abs(ilocPP[point] - ilocPP) > minSpace)[0]

            if len(ind) == 0:
                break

            ivalPP = ivalPP[ind]
            ilocPP = ilocPP[ind]

        return loc[0:k], val[0:k]

    def interpolateValues(
        self, magnitude, phase, N, cur_loc, cur_val, W, windowName, **winargs
    ):
        """Interpolate the peak locations, amplitudes, and phases
        magnitude := magnitude spectrum
        phase := phase spectrum
        N := size of spectrum
        cur_loc := peak locations
        cur_vals := peak amplitude values
        W := size of analysis window
        windowName := name of analysis window
        winargs := window-specific arguments"""

        eps = 0.0000000001

        iloc = np.zeros(len(cur_loc))
        iphase = np.zeros(len(cur_loc))
        iamp = np.zeros(len(cur_loc))

        zp = N / W
        wcoef = 1.0 / self.atomGenTable[windowName].coherentGain(W, **winargs)

        # parabolic interpolation of bin values
        leftval = magnitude[
            ((cur_loc - 1) * (cur_loc - 1 > -1) + (cur_loc - 1 <= -1 * 1)).astype(
                np.int16
            )
        ]  # tricky indexing hack, avoids edges
        rightval = magnitude[
            (
                (cur_loc + 1) * (cur_loc + 1 < N / 2) + (cur_loc + 1 >= N / 2 * N / 2)
            ).astype(np.int16)
        ]

        # in dB
        intpLV = 20 * np.log10(2 * (leftval + eps) / float(W) * wcoef)
        intpRV = 20 * np.log10(2 * (rightval + eps) / float(W) * wcoef)
        intpCV = 20 * np.log10(2 * (cur_val + eps) / float(W) * wcoef)

        iloc = cur_loc + 0.5 * (intpLV - intpRV) / (intpLV - 2 * intpCV + intpRV)
        iloc = ((iloc >= 0) & (iloc <= N / 2)) * iloc
        # remove nans!
        iloc[np.where(iloc != iloc)[0]] = 0.0

        # linear interpolation of phase
        leftphase = phase[np.floor(iloc).astype(np.int16)]
        rightphase = phase[np.floor(iloc).astype(np.int16) + 1]
        intpfactor = iloc - cur_loc
        intpfactor = (intpfactor > 0) * intpfactor + (intpfactor < 0) * (1 + intpfactor)
        diffphase = self.unwrap2pi(rightphase - leftphase)
        iphase = leftphase + intpfactor * diffphase

        # interpolation of amplitude
        ival = cur_val - 0.25 * (leftval - rightval) * (iloc - cur_loc)

        return iloc, ival, iphase
