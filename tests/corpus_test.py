import pydbm.data 
import pydbm.dictionary

import numpy as np
import matplotlib.pyplot as plt

corpus = '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/piano'

C = pydbm.data.Corpus(corpus)

list_of_corpora = ['/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/forte', '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/piano']
S = pydbm.data.SoundDatabase(list_of_corpora)

D = pydbm.dictionary.SoundgrainDictionary(S)
