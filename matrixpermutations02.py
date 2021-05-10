import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce


def blockMatrix(blocks):
    """creates a bloack 0-1 matrix.
    param blocks: list of non-negative integers which is the size of the blocks.
    a 0 block size corresponds to a 0 on the main diagonal.
    """
    blocks = np.array(blocks).astype("int64")
    f = lambda x: 1 if x == 0 else x
    n = np.sum([f(x) for x in blocks])
    n = int(n)
    A = np.zeros((n, n))
    pos = 0
    for i in range(len(blocks)):
        b = blocks[i]
        if b > 0:
            A[pos : pos + b, pos : pos + b] = np.ones((b, b))
        pos += f(b)
    return A


def permutationMatrix(ls):
    """returns a permutation matrix of size len(ls)^2.
    param ls: should be a reordering of range(len(ls)), which defines the
    permutation on the ROWS.
    returns a permutation matrix P.
    np.dot(P,A) should be rearrangement of the rows of A according to P.
    To permute the columns of a matrix A use:
    Q = np.transpose(P), then: np.dot(A,Q).
    """
    n = len(ls)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, ls[i]] = 1
    return P


def shuffleCopyMatrix(lins, louts, msize):
    """Returns a matrix P that represents switch and copy operations
    on the rows of a matrix.
    param msize: the size (of the square matrix).
    param lins: row indices to be replaced.
    param louts: row that replace the ones listed in lins.
    lins and louts must be of the same length and contain indiced within
    range(msize).
    These operations are performed on the identity matrix, and the result
    is the return value P.
    """
    # P = np.zeros((msize,msize))
    P = np.identity(msize)
    I = np.identity(msize)
    if not len(lins) == len(louts):
        return P
    for i in range(len(lins)):
        P[lins[i]] = I[louts[i]]
    return P


def scoreMatrix(n):
    """The score function of the matrix. The assumption is that the true
    arrangement maximized the interaction close to the main diagonal.
    The total sum of the interaction is an invariant, preserved by permuations.
    param n: size of ca 2-d n over n array.
    returns the score matrix, which is used to calculate the score of any given
    n^2 matrix.
    """
    s = np.arange(n)
    s = np.exp(-s)
    S = np.zeros((n, n))
    for i in range(n):
        S[i][i:] = s[: n - i]
    return S


def score(A, S):
    """returns the weighted sum (by the score matrix S) of
    the matrix A
    """
    return np.sum(A * S)

def constrainMatrix(ls, A):
    """Returns a matrix of the same dimension as A, but every entry with
    an index (either row or column) not in ls is 0.
    """
    B = np.zeros_like(A)
    #B[np.ix_(ls,ls)] = 1
    B[np.ix_(ls,ls)] = A[np.ix_(ls,ls)]
    #B[ls][:,ls] = A[ls][:,ls]
    return B

def resetIndices(ls, A):
    """essentially returns the constraint of A to the complement indices of ls,
    by reseting all the indices in ls to 0.
    """
    B = A.copy()
    B[ls,:] = 0
    B[:,ls] = 0
    return B


def reindexMatrix(iss, jss, A):
    """iss and jss are lists of indices of equal size, representing
    a permuation: iss[i] is replaced with jss[i]. all other indices which are
    not in the lists left unchanged.
    """
    n = len(A)
    B = np.zeros_like(A)
    tss = [i for i in range(n)]
    for i in range(len(iss)):
        tss[iss[i]] = jss[i]
    for i in range(n):
        for j in range(n):
            B[i, j] = A[tss[i], tss[j]]
    return B

# example
x = np.arange(25).reshape((5,5))
reindexMatrix([1,3],[3,1], x)

def scorePair(iss, jss, refmat, scoremat):
    A = np.zeros_like(refmat)
    l = iss + jss
    n = len(l)
    for i in range(n):
        for j in range(n):
            A[i, j] = refmat[l[i], l[j]]
    return score(A, scoremat)


def scorePair2(iss, jss, refmat):
    """scores the interaction of two segments
    iss and jss. weighted by the ideal diagonal distribution.
    """
    s = 0
    temp = 0
    for i in range(len(iss)):
        for j in range(len(jss)):
            temp = np.exp(-np.abs(j + len(iss) - i))
            # we only care about interaction between the 2 segments and not
            # inside each one of them which wouldn't be affected by
            # rearrangement.
            s += refmat[iss[i], jss[j]] * temp
    return s

def scorePair3(iss, jss, refmat, lreverse=False, rreverse=False):
    """iss, jss must be lists of segments of the index range of refmat,
    our reference matrix.
    reurns the interaction score of iss and jss as if reindexed the matrix so
    that they will be adjuscent to each other.
    """
    s = 0
    temp = 0
    for i in range(len(iss)):
        for j in range(len(jss)):
            x = iss[i]
            y = jss[j]
            if lreverse:
                x = iss[-1 - i]
            if rreverse:
                y = jss[-1 -j ]
            # temp = np.exp(-np.abs(i-j))
            #temp = np.exp(-np.abs(x - y))
            temp = np.exp(-np.abs(j + len(iss) - i))
            # we only care about interaction between the 2 segments and not
            # inside each one of them which wouldn't be affected by
            # rearrangement.
            s += refmat[x, y] * temp
    return s


# and the corrsponding indices are:
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
articulate([5,7,3])


plt.ion()




blocks = [4,4,4,4,4]
segs = [5,5,4,6]
articulate(blocks)
articulate(segs)

# the block matirx
X = blockMatrix(blocks)
X
plt.imshow(X)

ls = articulate(segs)
ls

scorePair2(ls[0],ls[0], X)
scorePair2(ls[0],ls[1], X)
scorePair2(ls[0],ls[2], X)
scorePair2(ls[0],ls[3], X)

plt.matshow( constrainMatrix(ls[0], X))
plt.matshow( constrainMatrix(ls[1], X))
plt.matshow( constrainMatrix(ls[2], X))
plt.matshow( constrainMatrix(ls[3], X))

# lets shuffle the segments
rs = articulate(segs)

# lets reverse segments 0 and 2
Y = blockMatrix(blocks)
Y

rs = articulate(segs)
rs[2] = np.flip(rs[2])
rs
#

Y = reindexMatrix(ls[2], rs[2], Y)

plt.matshow(X)
plt.matshow(Y)

rs[0] = np.flip(rs[0])
rs

Y = reindexMatrix(ls[0], rs[0], Y)

plt.matshow(X)
plt.matshow(Y)

plt.matshow( constrainMatrix(ls[0], Y))
plt.matshow( constrainMatrix(ls[1], Y))
plt.matshow( constrainMatrix(ls[2], Y))
plt.matshow( constrainMatrix(ls[3], Y))

## now lets change the orfer of the segments
neworder = [1,3,0,2]


newindices = [ls[i] for i in neworder]
newindices
newindices = reduce(lambda x,y: x + list(y), newindices, [])
newindices

Z = reindexMatrix(list(range(len(Y))), newindices, Y)

plt.matshow(X)
plt.matshow(Y)
plt.matshow(Z)

plt.matshow( constrainMatrix(ls[0], Z))
plt.matshow( constrainMatrix(ls[0], X))

plt.matshow( constrainMatrix(ls[1], Z))
plt.matshow( constrainMatrix(ls[2], Z))
plt.matshow( constrainMatrix(ls[3], Z))

newsegments = [len(ls[i]) for i in neworder]
newsegments
zs = articulate(newsegments)
zs

plt.matshow( constrainMatrix(ls[0], Z))
plt.matshow( constrainMatrix(zs[1], Z))
plt.matshow( constrainMatrix(zs[2], Z))
plt.matshow( constrainMatrix(zs[3], Z))


scorePair2(zs[0],zs[0], Z)
scorePair2(zs[0],zs[1], Z)
scorePair2(zs[0],zs[2], Z)
scorePair2(zs[0],zs[3], Z)

scorePair2(ls[1],ls[0], Y)
scorePair2(ls[1],ls[1], Y)
scorePair2(ls[1],ls[2], Y)
scorePair2(ls[1],ls[3], Y)

scorePair3(ls[2],ls[3], Y)
scorePair3(ls[2],ls[3], X)
scorePair3(ls[2],ls[3], Y, lreverse=True)


#def improve(A, xs):
#    temp = np.random.randint(1, len(xs))
#    iss = xs[0]
#    jss = xs[temp]
#    # we're going to see if iss and jss belong together in some configuration
#    sl = scorePair3(iss,jss, A)
#    slrv = scorePair3(iss,jss, A, lreverse=True)
#    sr = scorePair3(jss,iss, A)
#    srrv = scorePair3(jss,iss, A, rreverse=True)
#    t = np.max([sl, slrv, sr, srrv])
#    neworder = [len(x) for x in xs]
#    neworder[1] = len(xs[temp])
#    neworder[temp] = len(xs[1])
#    B = np.zeros_like(A)
#    if t == 0:
#        return xs, A
#    if t == sl:
#        newindices = [xs[i] for i in neworder]
#        newindices = reduce(lambda x,y: x + list(y), newindices, [])
#        B = reindexMatrix(list(range(len(A))), newindices, A)
#    elif t == sr:
#        p = neworder[0]
#        neworder[0] = neworder[1]
#        neworder[1] = p
#        newindices = [xs[i] for i in neworder]
#        newindices = reduce(lambda x,y: x + list(y), newindices, [])
#        B = reindexMatrix(list(range(len(A))), newindices, A)
#    elif t == slrv:
#        # first flip the segment 0
#        rs = xs.copy()
#        rs[0] = np.flip(rs[0])
#        B = reindexMatrix(xs[0], rs[0], A)
#        # then switch segment temp with 1
#        newindices = [xs[i] for i in neworder]
#        newindices = reduce(lambda x,y: x + list(y), newindices, [])
#        B = reindexMatrix(list(range(len(B))), newindices, B)
#    else:
#        # first flip the segment 0
#        rs = xs.copy()
#        rs[0] = np.flip(rs[0])
#        B = reindexMatrix(xs[0], rs[0], A)
#        # then make the switch
#        p = neworder[0]
#        neworder[0] = neworder[1]
#        neworder[1] = p
#        newindices = [xs[i] for i in neworder]
#        newindices = reduce(lambda x,y: x + list(y), newindices, [])
#        B = reindexMatrix(list(range(len(B))), newindices, B)
#    newsegs = [len(xs[i]) for i in neworder if i != 0]
#    newsegs[0] += len(xs[0])
#    newsegs = articulate(newsegs)
#    return newsegs, B



plt.matshow(Z)
zs

ws, W = improve(Z, zs)


# The basic idea is that we are given a matrix and its associated articulation,
# and we need to perform a possible reverese of on segmentm, and relocate
# a segment or two so they will be neighbors. we need to return the new
# articulation structure.

def flip1(s, A, arts):
    """flips (reverses) the s'th segment, as listed by arts.
    returns new matrix.
    param s: the segment to flip.
    param A: the matrix.
    param arts: the articulation of A.
    """
    myarts = arts.copy()
    myarts[s] = np.flip(myarts[s])
    B = reindexMatrix(arts[s], myarts[s], A)
    return B

# example
fooblocks = [4,6,2,7,11,6]
foo = blockMatrix(fooblocks)

foosegs = articulate([9, 15, 12])

plt.matshow(foo)
plt.matshow(constrainMatrix(foosegs[1], foo))
plt.matshow(constrainMatrix(foosegs[0], foo))

# flip segment 2
bar = flip1(1, foo, foosegs)
plt.matshow(foo)

plt.matshow(bar)

plt.matshow(constrainMatrix(foosegs[1], foo))
plt.matshow(constrainMatrix(foosegs[1], bar))

def indexing(arts):
   return reduce(lambda x,y: x + list(y), arts, [])

def swap2(s, r, A, arts):
    """swaps segments s and r, and returns the new
    matrix and the new segmentation.
    """
    myarts = arts.copy()
    myarts[s] = arts[r]
    myarts[r] = arts[s]
    B = reindexMatrix(indexing(arts), indexing(myarts), A)
    newarts = articulate([len(x) for x in myarts])
    return newarts, B

barsegs, bar = swap2(0,2, bar, foosegs)



def improve(A, xs):
    """param: A: matrix.
    param xs: associated articulation.
    checks if segments 0 and 1 should be attached in some particular order.
    """
    if len(xs) == 1:
        return xs, A
    iss = xs[0]
    jss = xs[1]
    # we're going to see if iss and jss belong together in some configuration
    sl = scorePair3(iss,jss, A)
    slrv = scorePair3(iss,jss, A, lreverse=True)
    sr = scorePair3(jss,iss, A)
    srrv = scorePair3(jss,iss, A, rreverse=True)
    t = np.max([sl, slrv, sr, srrv])
    mysegs = [len(xs[i]) for i in range(1, len(xs))]
    mysegs[0] += len(xs[0])
    mysegs = articulate(mysegs)
    if t == 0:
        return xs, A
    if t == sl:
        # nothin to change
        return mysegs, A
    elif t == sr:
        # swap 0 and 1 segments
        _, B = swap2(1,0, A, xs)
        return mysegs, B
    elif t == slrv:
        # first flip the segment 0
        B = flip1(0, A, xs)
        return mysegs, B
    else:
        # first flip the segment 0
        B = flip1(0, A, xs)
        # then make the switch
        _, B = swap2(1,0, B, xs)
        return mysegs, B


# tests

# example
fooblocks = [4,6,2,7,11,6]
foo = blockMatrix(fooblocks)

foosegs = articulate([9, 15, 12])

plt.matshow(foo)
plt.matshow(constrainMatrix(foosegs[1], foo))
plt.matshow(constrainMatrix(foosegs[0], foo))

# flip segment 2
bar = flip1(1, foo, foosegs)
plt.matshow(foo)
plt.matshow(bar)

plt.matshow(constrainMatrix(foosegs[1], foo))
plt.matshow(constrainMatrix(foosegs[1], bar))

barsegs, bar = improve(bar, foosegs)


plt.matshow(foo)

plt.matshow(bar)

barsegs, bar = improve(bar, barsegs)
plt.matshow(bar)

# now flip and replace
bar = flip1(0, foo, foosegs)
barsegs, bar = swap2(0,2, bar, foosegs)
plt.matshow(foo)
plt.matshow(bar)


barsegs, bar = improve(bar, barsegs)
plt.matshow(bar)

# a bigger experiment

fooblocks = [15,17,19,20,10,21,30,21,40,9,27,19]
foo = blockMatrix(fooblocks)

np.sum(fooblocks)

foosegs = [38, 34, 32, 38, 37, 34, 35]
np.sum(foosegs)
np.sum(fooblocks)

foosegs = articulate(foosegs)


plt.matshow(foo)

plt.matshow(constrainMatrix(foosegs[1], foo))
plt.matshow(constrainMatrix(foosegs[0], foo))

# now perform some flips and swaps
bar = flip1(1, foo, foosegs)
bar = flip1(5, foo, foosegs)
barsegs, bar = swap2(0,3, bar, foosegs)
barsegs, bar = swap2(1,2, bar, barsegs)
barsegs, bar = swap2(5,2, bar, barsegs)
barsegs, bar = swap2(5,0, bar, barsegs)
barsegs, bar = swap2(4,1, bar, barsegs)

plt.matshow(foo)
plt.matshow(bar)

while len(barsegs) > 1:
    x = np.random.randint(1,len(barsegs))
    barsegs, bar = swap2(1,x, bar, barsegs)
    barsegs, bar = improve(bar, barsegs)
plt.matshow(foo)
plt.matshow(bar)
# yeshhhhh!!!!!!!
