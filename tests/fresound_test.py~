import numpy as np
import urllib
import json
import xml.etree.ElementTree as etree

import audiolab
import pydbm_.dictionary
import pydbm_.data


#freesound stuff
low = 1
high = 20

api_key = '548e78fffaf14c169c473b2cd71b7440'
query = ['drop'] 
filters = ['duration:[* TO 0.5]', 'samplerate:44100', 'channels:1', 'type:wav']

P = urllib.URLopener()
outdir = '/Users/grahamboyes/Desktop/hackday'

for page in range(low, high):
    P.retrieve("http://www.freesound.org/api/sounds/search/?q=%s&f=%s&p=%s&api_key=%s&format=xml"%(''.join([q + ' ' for q in  query]), ''.join([f + ' ' for f in filters]), page, api_key), '%s/test.xml'%outdir)
    f = etree.parse('%s/test.xml'%outdir)
    root = f.getroot()
    for sound in root.findall('sounds'):
        for resource in sound.findall('resource'):
            P.retrieve(resource.find('serve').text + '?api_key=%s'%api_key, outdir + '/' + resource.find('original_filename').text)


#a decomposition using the outdir
x, fs, p = audiolab.wavread('/Users/grahamboyes/Documents/python/pythonDafx/SOUNDS/AMEN.wav')

S = pydbm_.data.SoundDatabase([outdir])

n = S.sdif2array('/Users/grahamboyes/Desktop/AMEN.mrk.sdif', ['XTRD'])['XTRD'] 
n = (n['time'] * fs).astype(int)

D = pydbm_.dictionary.SoundgrainDictionary(fs, S)
D.addCorpus(n, 0)

x_ = np.zeros(len(x) + max(D.atoms['duration']))
x_[0:len(x)] = x

mod, res, M = D.mp(x_.copy(), 300, 35)
audiolab.wavwrite(mod[0:len(x)], '%s/model.wav'%outdir, fs)
