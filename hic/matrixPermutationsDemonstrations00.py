# Here I tests shit but keep the functions somewhere else to make things tidy.
#from .mymodule.matrixpermutationsModule import *
#from mymodule import *

# this will not run if you try to execute the python file
# will throw an import module with unknow parent error
#from .foo import bar
#from .mymodule import *
#bar()

# on the other hand this will run
# but then pyright won't recognize it correctly and won'y show the function
# signature and the docstrings.
#from mymodule.matrixpermutationsModule import *
#import foo
#foo.bar()
#x = blockMatrix([10,10,10])
#plotMat(x, log(x+1))


# this seems to solve the issue for both
if __name__ == '__main__':
    from foo import bar
    from mymodule import *
else:
    from .foo import bar
    from .mymodule import *



bar()
x = blockMatrix([10,10])
P = permutationMatrix(list(concatv(range(4), range(15,20), range(10,15),
    range(4,10))))
s = scoreMatrix(len(x), logscore=False)
y = log(x+1)
w = log(x+1) + log(s) + 10
plotMat(x, P)
plotMat(y, log(s))
plotMat(y, y + log(s))

plotMat(y, P @ (y + log(s)) @ P.T)

plotMat(w, P @ w @ P.T)

plotMat(log(s), P @ w @ P.T)

plotMat(P @ log(s) @ P.T, P @ w @ P.T)

plotMat(P @ log(s) @ P.T, P @ log(s*x + 1) @ P.T)

plotMat(P @ x @ P.T , P @ w @ P.T)

plotMat(log(s), log(s) + log(x+1))
plotMat(log(s), log(s) + y)

plotMat(log(s), log(s) + y)
plotMat(P @ log(s) @ P.T , P @ (log(s) + y) @ P.T)

z = x*s + 1e-4
plotMat(z, log(z) + 10)

plotMat(log(z) + 10, P @ ( log(z) + 10 ) @ P.T)

l = list(concatv(range(4), range(15,20), range(10,15), range(4,10)))

plt.xticks(l)

plt.xticks(range(20), [str(x) for x in l])
plt.yticks(range(20), [str(x) for x in l])

myconv = np.zeros((4,4))
myconv[0] = [-0.5,-1,1,0.5]
myconv[1] = [-1,-2,2,1]
myconv[2] = [1,2,-2,-1]
myconv[3] = [0.5,1,-1,-0.5]

mymat = P @ ( log(z) + 10 ) @ P.T
mymatconv = convolve2d(P @ mymat @ P.T, -myconv, mode='same')
plotMat(mymat, mymatconv)



(list(concatv(range(4), range(15,20), range(15,20), range(4,10))))

l=list(concatv(range(4), range(15,20), range(10,15), range(4,10)))
l

x = blockMatrix([5,5])
P = permutationMatrix(
        list(
            concatv(range(2), range(8,10), range(5,8), range(2,5))
            ))
y = log(x+1)
s = scoreMatrix(len(x), logscore=True)

plotMat(y + s, P @ y + s @ P.T)

l = list(concatv(range(2), range(8, 10), range(5, 8), range(2, 5)))

z = np.zeros_like(x)
for i in range(len(x)):
    for j in range(len(x)):
        z[i,j] = x[l[i], l[j]]


A = np.arange(100).reshape((10,10))
P @ A @ P.T
np.diag(A)
np.diag(P @ A @ P.T)

w = log(x+1) + s + 10

plotMat(w, P @ w @ P.T)

plotMat(y + s, P @ (y + s) @ P.T)



myconv = np.zeros((4,4))
myconv[0] = [-1,-2,2,1]
myconv[1] = [-2,-3,3,2]
myconv[2] = [2,3,-3,-2]
myconv[3] = [1,2,-2,-1]
a = np.array([1,2,2,1,2,3,2,2,2,3,2,3,1,2,2,1]).reshape((4,4))
b = np.array([0,0.5,1,0.5,0,0.5,2,1,1,2,0.5,0,0.5,1,0.5,0]).reshape((4,4))
b

convolve2d(a,myconv,mode='same')
convolve2d(a,myconv,mode='valid')
convolve2d(b,myconv,mode='valid')

(myconv*a).sum()

myconv = np.zeros((4,4))
myconv[0] = [-1,-2,2,1]
myconv[1] = [-2,-3,3,2]
myconv[2] = [2,3,-3,-2]
myconv[3] = [1,2,-2,-1]
mymat = P @ ( log(z) + 10 ) @ P.T
mymatconv = convolve2d(P @ mymat @ P.T, -myconv, mode='valid')
plotMat(mymat, mymatconv)

temp = P @ mymat @ P

temp = temp[5:13,5:13]
convolve2d(temp, -myconv, mode='valid')








### Testing hi-c matrix

filepath = "./hicdata/191-98_hg19_no_hap_EBV_MAPQ30_merged.mcool"
bedpath = "./hicdata/191-98_reconstruction.bed"

filepath = "./hicdata/CL17-08_hg19_no_hap_EBV_MAPQ30_merged.mcool"
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

# loading cooler matrix with hy5's API
h5 = h5py.File(filepath, 'r')
h5.keys()
h5['resolutions'].keys()
h5['resolutions']['100000'].keys()
#h5['resolutions']['250000'].keys()
h5.close()

# loading cooler matrix with cooler's API
c = cooler.Cooler(filepath+"::resolutions/500000")
#c = cooler.Cooler(filepath+"::resolutions/250000")
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

def genToMatPos(df, sizedict, sizes):
    for row in df.iterrows():
        print(row[1][0])
genToMatPos(bedDF, d, c.chromsizes)

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

arr = c.matrix(balance='KR', sparse=False)[:,:]
arr = np.nan_to_num(arr)

c = cooler.Cooler(reffilepath+"::resolutions/500000")
refarr = c.matrix(balance='KR', sparse=False)[:,:]
refarr = np.nan_to_num(refarr)

def plotHicMat(arr, transform=np.log10):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.matshow(transform(arr), cmap='YlOrRd')
    fig.colorbar(im)

plotHicMat(arr+1)

plotHicMat(refarr+1)

plotHicMat(arr)

plotMat(arr, np.log(arr+1))

plotHicMat((arr+1) / (refarr+1))

plotHicMat((arr+1) / (0.3*refarr+1))

plotHicMat((arr+1) / (0.1*refarr+1))

n = len(arr)
s = scoreMatrix(n, logscore=True)

plotMat(s, np.log(arr+1) - s)

y = [arr[n-i,n-i] for i in range(1,n)]


a = c.matrix(balance='KR').fetch('2','5')

b = c.matrix(balance='KR').fetch('2','5')

plotHicMat(a)
