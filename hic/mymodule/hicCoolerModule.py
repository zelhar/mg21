import cooler
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from toolz.curried import interleave, reduce, concat, concatv
from toolz.curried import unique
# https://github.com/open2c/cooler-binder/blob/master/cooler_api.ipynb
from numpy import flip
from numpy import log
from scipy.signal import convolve2d

def addOffsetColumn(df, csize):
    """param df: a bed file with genomic ranges. first 3 coluns are chr, start,
    stop.
    param: csize: a list of the chromosomes' sizes.
    returns a column 'offset' which is the size of all chromosomes preceding
    chr.
    """
    #x = np.zeros_like(df['chr']).astype('int64')
    x = df['chr']
    y = [np.sum(csize[:c]) for c in x]
    return y

def dummy():
    """
    dummy function
    """
    print("hi")

