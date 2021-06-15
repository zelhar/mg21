import cooler
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

# https://github.com/open2c/cooler-binder/blob/master/cooler_api.ipynb

plt.ion()

filepath = "./hicdata/191-98_hg19_no_hap_EBV_MAPQ30_merged.mcool"

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


arr = c.matrix(balance=False, sparse=False)[:,:]
arr2 = c.matrix(balance='KR', sparse=False)[:,:]

arr

arr2 = np.nan_to_num(arr2)

np.allclose(arr, arr2, equal_nan=True)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
im = ax.matshow(np.log10(arr2), cmap='YlOrRd')
fig.colorbar(im)



