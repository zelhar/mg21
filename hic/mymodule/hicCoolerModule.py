import cooler
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from toolz.curried import interleave, reduce, concat, concatv
from toolz.curried import unique
from toolz.curried import compose, compose_left, comp, complement
from toolz.curried import pipe, thread_first, thread_last
from toolz.curried import first, second, nth
from toolz.curried import partial

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

def plotHicMat(arr, transform=np.log10):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.matshow(transform(arr), cmap='YlOrRd')
    fig.colorbar(im)

def getOffsets(chroms, cj):
    """
    chroms: a sorted list of chromnames, in oreder of chromosomes
    cj: a cooler object.
    returns a dictionary with the offset for each chromosome
    in a submatrix that contains only the listed chromosomes.
    """
    n = len(chroms)
    offset = {}
    offset[chroms[0]] = 0
    for i in range(1,n):
        offset[chroms[i]] = offset[chroms[i-1]] + cj.chromsizes[chroms[i-1]] // cj.binsize + 1
    return offset

def addOffsets(bedDF, offset):
    """
    bedDF: pandas data frame with genomic ranges.
    its first columns are: 'chr', 'start', 'end'.
    offset contains the offset of 'chr'.
    Returns a list which contains the correct offset for every row 
    of the bedDF (can be added as an extra column 'offset').
    """
    n = len(bedDF)
    x = np.zeros(n)
    for i in range(n):
        x[i] = offset[bedDF['chr'][i]]
    return x.astype('int64')


def getIntervals(bedDF, c):
    """
    bedDF is a data frame with the information about identified contiguous
    intervals in the hic data -- start, end and chromosome. 
    c is a cooler object for the same data.
    Will return a sorted list of the implied segments, including the chromosomes
    start and ends within the matrix. 
    Example for 2 chromosomes and given sections a,b in chr1 and c,d in chr2,
    the implied segmentation is: 
    chr1_start=0 <= a < b <= chr1_end < chr2_start <= c < d <= chr2_end
    """
    #incy = lambda x : (first(x), second(x) + 1) 
    #foo = compose(incy, c.extent)
    #l = map(incy, map(c.extent, c.chromnames))
    l = unique(concat(map(c.extent, c.chromnames[:-1])))
    l = sorted(list(l))
    for  i in range(len(bedDF)):
        #x = bedDF.iloc[i]['start'] // c.binsize + bedDF.iloc[i]['offset']
        #y = bedDF.iloc[i]['end'] // c.binsize  + bedDF.iloc[i]['offset']
        x = bedDF.iloc[i]['start'] // c.binsize + c.offset(bedDF.iloc[i]['chr'])
        y = bedDF.iloc[i]['end'] // c.binsize  + c.offset(bedDF.iloc[i]['chr'])
        if abs(y - c.extent(bedDF.iloc[i]['chr'])[1]) <=2:
            y = c.extent(bedDF.iloc[i]['chr'])[1]
        l.append(x)
        l.append(y)
    # we need to correct the boundries a bit
    #l = sorted(l, reverse=True)
    #return l
    return sorted(list(unique(l)))

#l = getIntervals(bedDF, c)
#l






