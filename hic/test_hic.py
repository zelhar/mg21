import straw
import numpy as np
from scipy.sparse import coo_matrix

import scipy.sparse as sparse

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import cm



#https://colab.research.google.com/drive/1548GgZe7ndeZseaIQ1YQxnB5rMZWSsSj

straw.straw?

res = 100000*5

spmat = straw.straw(
    "KR",
    "../../mnt/Yiftach_Kolb_project_hic_genome_reconstruction/191-98_hg19_no_hap_EBV_MAPQ30_merged.hic",
    "1", "1",
    unit="BP",
    binsize=res,
)


for i in range(10):
    print("{0}\t{1}\t{2}".format(spmat[0][i], spmat[1][i], spmat[2][i]))

n = np.max(spmat[0])
m = np.max(spmat[1])
n = max(n,m)
n

243199373 // res
#x = coo_matrix((spmat[2], (spmat[1], spmat[0])), shape=(n+1,n+1))


I = np.array(spmat[0][:])/res
J = np.array(spmat[1][:])/res
V = np.array(spmat[2][:])

sz=int(n/res)+1

M = coo_matrix((V,(I,J)),shape=(sz,sz))

#M = sparse.coo_matrix((V,(I,J)),shape=(sz,sz)).tocsr()

plt.ion()

x = M.toarray()
x[(np.isnan(x))] = 0

plt.matshow(np.log(x))

plt.colormaps()

plt.matshow(np.log10(x), cmap=cm.hot)

marks = np.zeros_like(x)
marks

plt.cla()

#marks = np.tri(sz, sz, -1)*500
#plt.matshow(np.log(marks))

marks = np.zeros(sz)
marks[192419497//res] = sz
marks[249250621//res] = sz

plt.plot(np.arange(sz), marks)

#plt.imshow(25500*np.log(x))
#plt.imshow(x)

plt.show()

plt.cla()


plt.close()

#sns.heatmap(np.log(x))



def getMatrixAsFlattenedVector(normalization, filepath, chrom, resolution, dozscore=False):
  for i in chrs:
    result = straw.straw(normalization, filepath, chrom, chrom, 'BP', resolution)
    I=np.array(result[0][:])/res
    J=np.array(result[1][:])/res
    V=np.array(result[2][:])
    sz=int(chr_sizes[str(i)]/res)+1
    M=sparse.coo_matrix((V,(I,J)),shape=(sz,sz)).tocsr()
    # make symmetric instead of upper triangular
    N=M+M.T-sparse.diags(M.diagonal(),dtype=int)
    A=N.reshape(1,sz*sz)
    if (i is not 1):
      vector = np.concatenate([vector, A.toarray().flatten()])
    else:
      vector = A.toarray().flatten()
    if dozscore:
      vector = stats.zscore(vector)
    return vector
