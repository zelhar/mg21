if __name__ == '__main__':
    from mymodule import *
else:
    from .mymodule import *


A = np.random.rand(10,8) * 1e-5

plt.matshow(A)


filepath1 = "./hicdata/191-98_hg19_no_hap_EBV_MAPQ30_merged.mcool"
bedpath = "./hicdata/191-98_reconstruction.bed"
filepath2 = "./hicdata/CL17-08_hg19_no_hap_EBV_MAPQ30_merged.mcool"
bedpath = "./hicdata/CL17-08_reconstruction.bed"

reffilepath = "./hicdata/CL18-38_hg19_no_hap_EBV_MAPQ30_merged.mcool"
# no bed it's the reference matrix

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


# loading cooler matrix with cooler's API
c = cooler.Cooler(filepath2 + "::resolutions/250000")

arr = c.matrix(balance='KR', sparse=False)[:,:]
arr = np.nan_to_num(arr)

c2 = cooler.Cooler(reffilepath+"::resolutions/500000")

refarr = c2.matrix(balance='KR', sparse=False)[:,:]
refarr = np.nan_to_num(refarr)

plotHicMat(arr+1)

plotHicMat(refarr+1)

# slicing by chromosomes
a2b5 = c.matrix(balance='KR').fetch('2', '5')
a2b5 = np.nan_to_num(a2b5)

plotHicMat(a2b5 + 1)


A = np.random.random((4,5))

A @ A.T

A = a2b5 @ a2b5.T
A = A + np.identity(len(A))
A

# column normalize:
A = A / A.sum(axis = 0)

x = np.ones((2,3))
x[0,1] = 3
x
y = x.T @ x
y

y
y.sum(axis=0)
y.sum(axis=1)

z = diffusionMatrix(y)
z.sum(axis=0)
z.sum(axis=1)

A.sum(axis=0)

K = diffusionMatrix(A)

K.sum(axis=0)

p = K @ np.ones(len(K))



plt.bar(range(len(K)), p)

mymax = np.argmax(p)
mymax


plotHicMat(a2b5)

plt.hlines(mymax, 0, 100, color='blue')

mymaxes = np.argsort(p)[-1:-10:-1]


plt.hlines(mymaxes, 0, 250, color='purple')


morelocations = np.argsort(p)[-1:-150:-1]

plt.hlines(morelocations, 0, 70, color='green')

plt.close()

plt.cla()

# loading cooler matrix with cooler's API
c = cooler.Cooler(filepath1 + "::resolutions/250000")
c2 = cooler.Cooler(filepath2 + "::resolutions/250000")

# slicing by chromosomes
a16b11 = c2.matrix(balance='KR').fetch('16', '11')
a16b11 = np.nan_to_num(a16b11)

plotHicMat(a16b11 + 1)

arr = c.matrix(balance='KR', sparse=False)[:,:]
arr = np.nan_to_num(arr)

arr2 = c2.matrix(balance='KR', sparse=False)[:,:]
arr2 = np.nan_to_num(arr2)

plotHicMat(arr +1)

plotHicMat(arr2 +1)

x = c2.matrix(balance='KR').fetch('2', '11')
x = np.nan_to_num(x)
plotHicMat(x + 1)
