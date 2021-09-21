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
#plt.ion()

filepath = "./hicdata/191-98_hg19_no_hap_EBV_MAPQ30_merged.mcool"
bedpath = "./hicdata/191-98_reconstruction.bed"

filepath = "./hicdata/CL17-08_hg19_no_hap_EBV_MAPQ30_merged.mcool"
bedpath = "./hicdata/CL17-08_reconstruction.bed"

bedDF = pd.read_csv(
    bedpath,
    names=[
        "chr",
        "start",
        "end",
        "foo",
        "bar",
        "orientation",
        "derivative_chr",
        "scaffold",
    ],
    sep="\t",
)
bedDF


# loading cooler matrix with hy5's API
h5 = h5py.File(filepath, 'r')

h5.keys()
h5['resolutions'].keys()
h5['resolutions']['250000'].keys()

l = list(h5['resolutions']['250000'].keys())
l

h55=h5['resolutions']['250000']

h55['pixels'].keys()

h55['pixels']['count'][:10]

h5.close()



# loading cooler matrix with cooler's API
c = cooler.Cooler(filepath+"::resolutions/250000")

binsize = c.info['bin-size']
binsize

# dictionary of name:size
d = zip(c.chromnames, c.chromsizes)
d = dict(d)
d

# dictionary of name:0-index
d2 = zip(c.chromnames, range(25))
d2 = dict(d2)
d2

df['offset'] = 0

def genToMatPos(df, sizedict, sizes):
    for row in df.iterrows():
        print(row[1][0])
genToMatPos(bedDF, d, c.chromsizes)

cooler.fileops.is_multires_file(filepath)

c.info

c.chroms()[:]
c.chromnames
c.chromsizes


c.bins()[:10]

bins = c.bins()[['chrom', 'start', 'end']]
bins[:10]

len(bins[:])

c.pixels(join=True)[:10]
c.pixels()[:10]


#arr = c.matrix(balance=False, sparse=False)[:,:]
arr2 = c.matrix(balance='KR', sparse=False)[:,:]
#arr
arr2 = np.nan_to_num(arr2)
#np.allclose(arr, arr2, equal_nan=True)

# slicing by chromosomes
a2b5 = c.matrix(balance='KR').fetch('2', '5')

A4 = c.matrix(balance='KR', sparse=True).fetch('2', '3')
A4.toarray()


plt.matshow(np.log(a2b5), cmap='YlOrRd')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
im = ax.matshow(np.log10(arr2), cmap='YlOrRd')
fig.colorbar(im)

arr3 = 20*arr2 + 1.1
arr4 = np.log10(arr3)

w = np.zeros((5,5))
w[1] = [0,0,-1,0,0]
w[2] = [0,-1,4,-1,0]
w[3] = [0,0,-1,0,0]

w[1:4,1:4]=-1
w[2,2] = 5

z = convolve2d(arr4, -w, mode='same')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
im = ax.matshow(z, cmap='YlOrRd')
fig.colorbar(im)


x = np.arange(12000)
plt.plot(x,x)

offset = c.chromsizes[:3].sum()
offset 

x = np.arange(18989850 // binsize,	72612732//binsize) + (offset // binsize)

y = np.ones_like(x)

plt.plot(y,x)

x = np.arange(12000)

y = ones_like(x)

y *= 2836


plt.plot(x,y)

plt.plot(y,x)

72612732 // binsize

plt.cla()

def articulate(l):
    """l is a list of positive integers.
    returns the implied articulation, meaning a list of lists (or 1d arrays)
    ls, such that ls[0] it the numbers 0 to l[0]-1, ls[1] is a list of the
    numbers ls[1] to ls[2]-1 etc.
    """
    # ls = [np.arange(l[0]).astype('uint64')]
    ls = []
    offsets = np.cumsum([0] + l)
    for i in range(0, len(l)):
        xs = np.arange(l[i]).astype("uint64") + offsets[i]
        ls.append(xs)
    return ls

# example:
articulate([5, 7, 3])

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

bedDF['offset'] = addOffsetColumn(bedDF, c.chromsizes)
bedDF

bedDF['absStart'] = bedDF['start'] + bedDF['offset']
bedDF['absEnd'] = bedDF['end'] + bedDF['offset']
#bedDF['absEnd'] = bedDF['end'] + bedDF['offset'] + binsize
bedDF

n = c.chromsizes.sum()

ls = interleave([bedDF['absStart'] // binsize , bedDF['absEnd'] // binsize + 1 ])
ls = list(ls)
ls
#ls =  [x // binsize for x in ls]
#ls
ls = sorted(ls)
ls
ls.insert(0,0)
ls.append(n // binsize)
ls


ls = unique(ls)
ls = list(ls)
ls

x = np.array(ls)
y = np.ones_like(ls) * 10000

plt.scatter(x,x, color='green')

plt.scatter(bedDF['absStart'] // binsize,bedDF['absEnd'] // binsize + 1, color='green')

z = c.chromsizes // binsize
z = np.cumsum(z)

plt.scatter(z,z, color='blue')

plt.vlines(ls, ymin=0, ymax=12000, color='green')

plt.hlines(ls, 0, 12000, color='green')

plt.vlines(bedDF['absStart'] // binsize, ymin=0, ymax=12000, color='green')

plt.hlines(bedDF['absStart'] // binsize, 0, 12000, color='green')

plt.vlines(bedDF['absEnd'] // binsize, ymin=0, ymax=12000, color='blue')

plt.hlines(bedDF['absEnd'] // binsize, 0, 12000, color='blue')

plt.close()

plt.cla()

