import straw
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, tril, triu, hstack, vstack
from matplotlib import cm

# effectively a local chrom.sizes file for hg19
# chrs=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','Y')
chr_sizes = {
    "1": 249250621,
    "2": 243199373,
    "3": 198022430,
    "4": 191154276,
    "5": 180915260,
    "6": 171115067,
    "7": 159138663,
    "8": 146364022,
    "9": 141213431,
    "10": 135534747,
    "11": 135006516,
    "12": 133851895,
    "13": 115169878,
    "14": 107349540,
    "15": 102531392,
    "16": 90354753,
    "17": 81195210,
    "18": 78077248,
    "19": 59128983,
    "20": 63025520,
    "21": 48129895,
    "22": 51304566,
    "X": 155270560,
    "Y": 59373566,
}



def getOneMatrix(hicfile, res, chr1, chr2, norm="KR", unit="BP"):
    """This is a wraper that calls straw.straw (same input params), and then
    produces a coo_matrix out of it.
    """
    spmat = straw.straw(
        norm,
        hicfile,
        chr1,
        chr2,
        unit,
        res,
    )
    n = np.max(spmat[0]) #rows
    m = np.max(spmat[1]) #columns
    I = np.array(spmat[0][:])/res
    J = np.array(spmat[1][:])/res
    V = np.array(spmat[2][:])
    szr=int(n/res)+1
    szc=int(m/res)+1
    M = coo_matrix((V,(I,J)),shape=(szr,szc))
    return M


def getMatrices(hicfile, ls , res, norm="KR", unit="BP"):
    """
    Basically ls is a list of indices (chromosomes)
    returns a dictionary so that Ms[i,j] is chri vs chrj matrix (coo_matrix)
    """
    ls.sort()
    Ms = {}
    for i in ls:
        for j in ls:
            if i<j:
                Ms[i,j] = getOneMatrix(hicfile, res, str(i), str(j), norm, unit)
            elif i==j:
                Ms[i,j] = getOneMatrix(hicfile, res, str(i), str(j), norm, unit)
                Ms[i,j] = Ms[i,j] + triu(Ms[i,j], 1).transpose()
            else: #i>j
                Ms[i,j] = Ms[j,i].transpose()
    return Ms

def joinMatrices(hicfile, ls , res, norm="KR", unit="BP"):
    """
    Joins the output of getMatrices (a dictionary of matrices)
    into the complete contact matrix.
    """
    ls.sort()
    Ms = getMatrices(hicfile, ls, res, norm, unit)
    M = hstack([Ms[ls[0],i] for i in ls])
    for j in ls[1:]:
        temp = hstack([Ms[j,i] for i in ls])
        M = vstack([M,temp])
    return M
    
def cleanMatrix(x, v):



myhic = "./hicdata/CL18-38_hg19_no_hap_EBV_MAPQ30_merged.hic"

hic1 = "./hicdata/191-98_hg19_no_hap_EBV_MAPQ30_merged.hic"
hic2 = "./hicdata/CL17-08_hg19_no_hap_EBV_MAPQ30_merged.hic"
hic3 = "./hicdata/CL18-38_hg19_no_hap_EBV_MAPQ30_merged.hic"

M = getOneMatrix(myhic, 500000, "1", "2", "KR", "BP")
x = M.toarray()
x.shape
x[(np.isnan(x))] = 0

plt.ion()

plt.matshow(np.log(x))

M11 = getOneMatrix(myhic, 500000, "1", "1", "KR", "BP")
M12 = getOneMatrix(myhic, 500000, "1", "5", "KR", "BP")
M21 = getOneMatrix(myhic, 500000, "5", "1", "KR", "BP")
M22 = getOneMatrix(myhic, 500000, "5", "5", "KR", "BP")

M11.shape
M12.shape #same as M21
M21.shape
M22.shape

M1 = sparse.hstack([M11, M12])

M2 = sparse.vstack([M21, M22])
M2 = M2.transpose()
M2.shape

M = sparse.vstack([M1, M2])

x = M.toarray()
x.shape
x[(np.isnan(x))] = 0
plt.matshow(np.log(x))

z = M11 + triu(M11, 1).transpose()
z = z.toarray()
plt.matshow(np.log(z))

Ms = getMatrices(myhic, [1, 2, 3, 4], 500000, "KR", "BP")

M = joinMatrices(myhic, [i for i in range(1,6)], 500000, "KR", "BP")

M = joinMatrices(hic1, [i for i in range(1,6)], 500000, "KR", "BP")
M = joinMatrices(hic2, [i for i in range(1,6)], 500000, "KR", "BP")

M = joinMatrices(hic3, [i for i in range(1,6)], 500000, "KR", "BP")

x = M.toarray()
x[(np.isnan(x))] = 0
plt.ion()
plt.matshow(np.log10(x), cmap=cm.hot)

#y = np.argsort(x.flatten())
y = np.sort(x.flatten())
cut = y[-len(x)]
cut

z = x * (x >= cut/1000)

plt.matshow(np.log10(z), cmap=cm.hot)
