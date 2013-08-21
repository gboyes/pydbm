import pysdif
import numpy as np
import scikits.audiolab as audiolab
import os
import sys
import scipy.signal as sig
import time

f = pysdif.SdifFile('testSDIF5.sdif', 'r')
s = f.get_stream_IDs()
resample = 2

#sund grain dictionary
S = {}
ravg = 0.
favg = 0.
for i, sg in enumerate(s):
    S[i+1] = {}
 
    t = time.time()
    if os.path.splitext(sg.source)[1] == '.aif':
        x, fs, p = audiolab.aiffread(sg.source)
        if len(np.shape(x)) > 1:
            x = x[:, 0]
        x_ = sig.resample(x, len(x)/resample)
        S[i+1]['signal'] = x_ 

    elif os.path.splitext(sg.source)[1] == '.wav':
        x, fs, p = audiolab.wavread(sg.source)
        if len(np.shape(x)) > 1:
            x = x[:, 0]
        x_ = sig.resample(x, len(x)/resample)
        S[i+1]['signal'] = x_
    ravg+=time.time()-t

    #build the descriptor object
    t = time.time()
    for ft in f.get_frame_types():
         S[i+1][ft.signature]  = {}
         for component in ft.components:
             S[i+1][ft.signature][component.signature] = {}    
             S[i+1][ft.signature][component.signature]['name'] = component.name
             S[i+1][ft.signature][component.signature]['num'] = component.num
             S[i+1][ft.signature][component.signature]['values'] = []
             S[i+1][ft.signature][component.signature]['time'] = []
    favg+=time.time() - t

ti = time.time()
for frame in f:
    for matrix in frame:
        S[frame.id][frame.signature][matrix.signature]['values'].append((matrix.get_data()[0]))
        S[frame.id][frame.signature][matrix.signature]['time'].append(frame.time)

for i, sg in enumerate(s):
    for key in S[i+1].keys():
        if key != 'signal':
            for skey in S[i+1][key]:
                time_ = np.zeros(len(S[i+1][key][skey]['time']))
                if len(time_) > 0:
                    values = np.zeros((np.shape(S[i+1][key][skey]['values'][0])[0], len(S[i+1][key][skey]['values'])))
                    for t in range(len(S[i+1][key][skey]['time'])):
                        time_[t] = S[i+1][key][skey]['time'][t]
                        for vt in range(np.shape(S[i+1][key][skey]['values'][t])[0]):
                            values[vt, t] = S[i+1][key][skey]['values'][t][vt]
                    S[i+1][key][skey]['time'] = time_
                    S[i+1][key][skey]['values'] = values
             

print(ravg/len(s), favg/len(s), time.time() - ti)
