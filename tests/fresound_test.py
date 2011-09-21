import numpy as np
import urllib
import json
import xml.etree.ElementTree as etree

import audiolab
import pydbm_.dictionary
import pydbm_.data

kp = 5
fs = 44100

api_key = '548e78fffaf14c169c473b2cd71b7440'

query = ['voice'] 
filters = ['duration:[* TO 1]', 'samplerate:44100', 'channels:1', 'type:wav']


P = urllib.URLopener()
outdir = '/Users/grahamboyes/Desktop/freesound_tmp'

for page in range(1, 5):

    P.retrieve("http://www.freesound.org/api/sounds/search/?q=%s&f=%s&p=%s&api_key=%s&format=xml"%(''.join([q + ' ' for q in  query]), ''.join([f + ' ' for f in filters]), page, api_key), '%s/test.xml'%outdir)

    f = etree.parse('%s/test.xml'%outdir)
    root = f.getroot()

    for sound in root.findall('sounds'):
        for resource in sound.findall('resource'):
            P.retrieve(resource.find('serve').text + '?api_key=%s'%api_key, outdir + '/' + resource.find('original_filename').text)


#a decomposition
x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/python/pythonDafx/SOUNDS/AMEN.wav')

S = pydbm_.data.SoundDatabase([outdir])

n = S.sdif2array('/Users/grahamboyes/Desktop/AMEN.mrk.sdif', ['XTRD'])['XTRD'] 
n = (n['time'] * fs).astype(int)

D = pydbm_.dictionary.SoundgrainDictionary(fs, S)
#n = np.arange(0, len(x), 1024)

D.addCorpus(n, 0)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

mod, res, M = D.mp(x_.copy(), 180, 35)
audiolab.wavwrite(mod, '%s/prawn.wav'%outdir, fs)
