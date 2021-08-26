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



x = blockMatrix([200,150,150,200])
n = len(x)
s = scoreMatrix(n, logscore=False)

plotMat(x,s)

plotMat(np.log(x+1) ,np.log(s*(x*100 + 1e-5)))

plotHicMat(1e-2*s*(100*x+1e-9), transform=np.log)

l = translocationPermutation(219, 270, 540, n)
l = translocationPermutation(400, 450, 100, n)
P = permutationMatrix(l)
z = 1e-2*s*(100*x+1e-9)
z = np.log(z)
y = P @ z @ P.T

plotMat(z,y)

plotHicMat(y, transform= lambda x: x)
plotHicMat(z, transform= lambda x: x)

l = translocationPermutation(229, 270, 540, n)
P = permutationMatrix(l)
z = 1e-2*s*(100*x+1e-9)
z = np.log(z)
y = P @ z @ P.T
plotHicMat(y, transform= lambda x: x)

l2 = translocationPermutation(520, 530, 50, n)
Q = permutationMatrix(l2)

P = permutationMatrix(l)

z = 1e-2*s*(100*x+1e-9)
z = np.log(z)
y = P @ z @ P.T
plotMat(z,y)

plotMat(z,Q @ y @ Q.T)

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
#bedDF
bedDF['chr'] = bedDF['chr'].astype('str')
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
offsets = getOffsets(c.chromnames, c)

bedDF['offset'] = addOffsets(bedDF, offsets)

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

c2 = cooler.Cooler(reffilepath+"::resolutions/500000")

refarr = c2.matrix(balance='KR', sparse=False)[:,:]
refarr = np.nan_to_num(refarr)

plotHicMat(arr+1)

plotHicMat(refarr+1)

plotHicMat(arr)

plotMat(arr, np.log(arr+1))

plotHicMat((arr+1) / (refarr+1))

plotHicMat((arr+1) / (0.3*refarr+1))

plotHicMat((arr+1) / (0.1*refarr+1))

plotHicMat((arr+1) / (0.7*refarr+0.7))

plotHicMat(np.exp(leftGradientMatrix(np.log(arr))))

# convolutions

sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

derk = np.zeros((5,5))
derk[0] = [2,2,4,2,2]
derk[1] = [1,1,2,1,1]
derk[3] = - derk[1]
derk[4] = - derk[0]
derk

mymatconv = convolve2d(arr, sobel.T, mode='valid')
plotHicMat(mymatconv+1)

mymatconv = convolve2d(arr, derk, mode='valid')
plotHicMat(mymatconv+1)

## the interval, blocks and articualtion
l = getIntervals(bedDF, c)
l
blocks = [l[i+1] - l[i] for i in range(0,len(l)-1)]
blocks

foo = arr.copy()
foosegs = articulate(blocks)

scorePair3(foosegs[0], foosegs[1], foo)

scorePair3(foosegs[0], foosegs[2], foo)
scorePair3(foosegs[0], foosegs[3], foo)
scorePair3(foosegs[6], foosegs[13], foo)

B = flip1(0, foo, foosegs)
plotHicMat(B+1)

barsegs, bar = swap2(0, 10, foo, foosegs)

plotHicMat(bar+1)
#while len(foosegs) > 1:
#    x = np.random.randint(1, len(foosegs))
#    foosegs, foo = swap2(1, x, foo, foosegs)
#    foosegs, foo = improve(foo, foosegs)
#
#plt.matshow(foo)
#plt.matshow(bar)

def findNeighbor(A, arts):
    """
    Finds the nearest segment to segment arts[0]
    returns the index of the neighbor, the score, and the
    orientation (encoded) which yields the best score.
    """
    best = 0
    argbest = 0
    orientation = 0
    for i in range(1, len(arts)):
        sl = scorePair3(arts[0], arts[i], A)
        slrv = scorePair3(arts[0], arts[i] , A, lreverse=True)
        sr = scorePair3(arts[i], arts[0], A)
        srrv = scorePair3(arts[i], arts[0] , A, lreverse=True)
        t = np.max([sl, slrv, sr, srrv])
        if t > best:
            best = t
            argbest = i
            orientation = np.argmax([sl, slrv, sr, srrv])
    return argbest, best, orientation

def joinFirst(A, arts):
    """
    find the nearest neighbor to the first segment in terms of 
    scor. Then combine them into one segment in the correct orientation.
    return the new matrix and its segmentation.
    """
    argbest, best, orientation = findNeighbor(A, arts)
    barts, B = swap2(1, argbest, A, arts)
    mysegs = [len(barts[i]) for i in range(1, len(arts))]
    mysegs[0] += len(barts[0])
    mysegs = articulate(mysegs)
    if orientation == 1:
        flip1(0, B, barts)
    elif orientation == 2:
        flip1(0, B, barts)
        flip1(1, B, barts)
    elif orientation == 3:
        flip1(1, B, barts)
    return mysegs, B 

def quickScore(A, arts, x, y, l):
    """
    scores of segments x and y based on the first or last
    l indiced only.
    """
    xs = arts[x][-l:]
    ys = arts[y][:l]
    return np.sum(A[xs][:,ys])




foo = arr.copy()
foo = (arr+1) / (0.3*refarr+1)
foosegs = articulate(blocks)

j = 6
for i in range(j+1, len(blocks)):
    print(quickScore(foo, articulate(blocks), j, i, 6))

findNeighbor(foo, foosegs)

len(foosegs)

while len(foosegs) > 1:
    foosegs, foo = joinFirst(foo, foosegs)
    pipe(foosegs, len, print)

plotHicMat(foo)

plotHicMat((arr+1) / (0.3*refarr+1))

plotHicMat(arr)

## trying stuff on partial arrays

n = len(arr)
s = scoreMatrix(n, logscore=True)

plotMat(s, np.log(arr+1) - s)

y = [arr[n-i,n-i] for i in range(1,n)]

myrange = np.array(list(concatv(
    range(c.offset('1') , c.offset('2')),   # chr1
    range(c.offset('3'), c.offset('5')),    # chr3,4
    range(c.offset('11'), c.offset('12')),  # chr 11
    range(c.offset('12'), c.offset('13')),  # chr 13
    )))

a = arr[myrange[:, np.newaxis], myrange ]
b = refarr[myrange[:, np.newaxis], myrange ]

plotHicMat(convolve2d(a, derk.T, mode='valid')+1)

plotHicMat(b+1)

plotHicMat(a+1)
plt.xticks(np.array([200, 700, 1100, 1450, 1700]), ['ch1', 'ch3', 'chr4', 'chr11', 'chr13'])

#locs, labels = plt.xticks()
plotHicMat((a+1) / (0.2*b + 1) )
plt.xticks(np.array([200, 700, 1100, 1450, 1700]), ['ch1', 'ch3', 'chr4', 'chr11', 'chr13'])

offset = {}
offset['1'] = 0
#offset['3'] = c.offset('2')
offset['3'] = offset['1'] + c.chromsizes['1'] // binsize + 1
offset['4'] = offset['3'] + c.chromsizes['3'] // binsize + 1
offset['11'] = offset['4'] + c.chromsizes['4'] // binsize + 1
offset['13'] = offset['11'] + c.chromsizes['11'] // binsize + 1

offset = getOffsets(['1','3','4','11','13'], c)
offset

x = addOffsets(bedDF, offset)
bedDF['offset'] = x
bedDF





### more partial arrays

a = c.matrix(balance='KR').fetch('2','5')

b = c2.matrix(balance='KR').fetch('2','5')

myrange = np.array(list(concatv(
    range(c.offset('2') , c.offset('5')), range(c.offset('11'), c.offset('12'))
    )))

a = arr[myrange[:, np.newaxis], myrange ]
b = refarr[myrange[:, np.newaxis], myrange ]

plotHicMat(a)

plotHicMat((a+1) / (0.2*b + 1) )

plotHicMat((a+1) / (0.5*b + 1) )

plotHicMat((a+1) / (1.5*b + 1) )

plotHicMat((a+1) / (10.5*b + 1) )

offset2 = 0
# start of chr3 in the sub-matrix:
offset3 = c.offset('3') - c.offset('2')
# start of chr4 in the sub-matrix:
offset4 = offset3 + c.offset('4') - c.offset('3')
# start of chr11 is end of 4 in the sub-matrix:
offset11 = offset4 + c.offset('5') - c.offset('4')

n = len(a)
y = np.arange(n)
x = np.ones_like(y)

plt.plot(offset3 * x, y, color='black')
plt.plot(offset4 * x, y, color='black')
plt.plot(offset11 * x, y, color='black')
plt.plot(y,offset3 * x,  color='black')
plt.plot(y,offset4 * x,  color='black')
plt.plot(y,offset11 * x,  color='black')

#locations = bedDF[bedDF.chr == 3]['end']
#locations

loc2 = bedDF[bedDF.chr == 2][['start', 'end']].to_numpy()
loc11 = bedDF[bedDF.chr == 11][['start', 'end']].to_numpy()

#plt.plot((offset3 + locations[2] // binsize) * x, y, color='red')
#plt.plot((offset3 + locations[7] // binsize) * x, y, color='red')
plt.plot((offset11 + loc11[0][0] // binsize) * x, y, color='red')
plt.plot((offset11 + loc11[0][1] // binsize) * x, y, color='red')

plt.plot((offset11 + loc11[11][0] // binsize) * x, y, color='purple')
plt.plot((offset11 + loc11[11][1] // binsize) * x, y, color='brown')

plt.plot((offset11 + loc11[12][0] // binsize) * x, y, color='purple')
plt.plot((offset11 + loc11[12][1] // binsize) * x, y, color='brown')

# big one
plt.plot(y, (offset11 + loc11[12][0] // binsize) * x, color='purple')
plt.plot(y, (offset11 + loc11[12][1] // binsize) * x, color='brown')

plt.plot((offset11 + loc11[12][0] // binsize) * x, y, color='purple')
plt.plot((offset11 + loc11[12][1] // binsize) * x, y, color='brown')


plt.close()

