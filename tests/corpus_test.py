import pydbm_.data 
import pydbm_.dictionary

import numpy as np
#import matplotlib.pyplot as plt

corpus = '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/piano'

C = pydbm_.data.Corpus(corpus)

list_of_corpora = ['/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/forte', '/Users/grahamboyes/Documents/Work/project_m/16khz/ordinario/SoftMallet/piano']
S = pydbm_.data.SoundDatabase(list_of_corpora)

D = pydbm_.dictionary.SoundgrainDictionary(S)