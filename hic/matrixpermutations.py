import numpy as np

# import torch
from PIL import Image
import matplotlib.pyplot as plt

from functools import reduce


A = np.identity(4)
A

P = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
P
Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Q


B = np.arange(16).reshape((4, 4))
B

B
np.dot(B, P)
np.dot(P, np.dot(B, P))


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
    return np.sum(A * S)


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
    print(tss)
    for i in range(n):
        for j in range(n):
            B[i, j] = A[tss[i], tss[j]]
    return B


reindexMatrix([1, 5], [5, 1], np.arange(36).reshape((6, 6)))


X = permutationMatrix([0, 4, 3, 2, 5, 1])
Y = np.transpose(X)

S = shuffleCopyMatrix([1, 3], [0, 2], 4)
S

T = np.transpose(S)
T

R = shuffleCopyMatrix([0, 1, 2, 3], [2, 0, 3, 1], 4)
R

blockMatrix([2, 3])

blockMatrix([2, 0, 0, 3])


blocks = [1, 3, 0, 3]


np.random.shuffle(B)


Z = blockMatrix([10, 20, 0, 0, 10, 20, 30]).astype("int64")
Z
ZZ = 255.0 * Z

im = Image.fromarray(ZZ)

im.show()

plt.ion()
# plt.ioff()


plt.imshow(ZZ)

plt.imshow(im)

plt.matshow(ZZ)

plt.close()

# ls = [10,20,0,0,10,20,30]
l1 = [i for i in range(25)]
l2 = [i + 25 for i in range(27)]
l3 = [i + 25 + 27 for i in range(20)]
l4 = [i + 25 + 27 + 20 for i in range(20)]

l3b = l3.copy()
l3b.reverse()
l3b

ZZ.shape

# rows
PP1 = permutationMatrix(l3 + l1 + l2 + l4)
PP2 = permutationMatrix(l1 + l2 + l3b + l4)
PP3 = permutationMatrix(l1 + l3b + l2 + l4)
# columns
QQ1 = np.transpose(PP1)  # then: np.dot(A,QQ).
QQ2 = np.transpose(PP2)  # then: np.dot(A,QQ).
QQ3 = np.transpose(PP3)  # then: np.dot(A,QQ).
ZZZ1 = np.dot(np.dot(PP1, ZZ), QQ1)
ZZZ2 = np.dot(np.dot(PP2, ZZ), QQ2)
ZZZ3 = np.dot(np.dot(PP3, ZZ), QQ3)

# plt.imshow(ZZZ)
# plt.imshow(ZZ)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax1.imshow(ZZ)
ax2.imshow(ZZZ)

fig, axs = plt.subplots(nrows=2, ncols=2)

fig.suptitle("original pattern and permutations")

axs[0, 0].imshow(ZZ)
axs[0, 0].set_title("original")

axs[0, 1].imshow(ZZZ1)
axs[0, 1].set_title("[52:73] moved to the start")

axs[1, 0].imshow(ZZZ2)
axs[1, 0].set_title("[52:73] reversed")

axs[1, 1].imshow(ZZZ3)
axs[1, 1].set_title("[52:73] moved to [25:52] and reversed")

plt.close()


sm = scoreMatrix(len(ZZ))
sm

score(ZZ, sm)
score(ZZZ1, sm)
score(ZZZ2, sm)
score(ZZZ3, sm)


def scorePair(iss, jss, refmat, scoremat):
    A = np.zeros_like(refmat)
    l = iss + jss
    n = len(l)
    for i in range(n):
        for j in range(n):
            A[i, j] = refmat[l[i], l[j]]
    return score(A, scoremat)


[scorePair(l2, l4, ZZ, sm)]

np.argmax([1, 9, 0, 3])

# reassembl
cs = [l2, l4, l1, l3]
cs

while len(cs) > 1:
    xs = cs.pop()
    l = np.argmax([scorePair(xs, y, ZZ, sm) for y in cs])
    sl = scorePair(xs, cs[l], ZZ, sm)
    r = np.argmax([scorePair(y, xs, ZZ, sm) for y in cs])
    sr = scorePair(cs[r], xs, ZZ, sm)
    if sl > sr:
        cs[l] = xs + cs[l]
    else:
        cs[r] = cs[r] + xs
    print(l, sl, r, sr)


test = cs[0]
test == l1 + l2 + l3 + l4


def scorePair2(iss, jss, refmat):
    s = 0
    temp = 0
    for i in range(len(iss)):
        for j in range(len(jss)):
            temp = np.exp(-np.abs(i - j))
            # we only care about interaction between the 2 segments and not
            # inside each one of them which wouldn't be affected by
            # rearrangement.
            s += refmat[iss[i], jss[j]] * temp
    return s


# reassembly 2
cs = [l2, l4, l1, l3]
cs

while len(cs) > 1:
    xs = cs.pop()
    l = np.argmax([scorePair2(xs, y, ZZ) for y in cs])
    sl = scorePair2(xs, cs[l], ZZ)
    r = np.argmax([scorePair2(y, xs, ZZ) for y in cs])
    sr = scorePair2(cs[r], xs, ZZ)
    if sl > sr:
        cs[l] = xs + cs[l]
    else:
        cs[r] = cs[r] + xs
    print(l, sl, r, sr)


test == l1 + l2 + l3 + l4

myblocks = [10, 15, 17, 19, 17, 15, 10]
mymatrix = blockMatrix(myblocks)

dmatrix = scoreMatrix(len(mymatrix))

dmatrix += np.transpose(dmatrix)

dmatrix -= np.identity(len(dmatrix))

plt.matshow(mymatrix)

plt.matshow(np.log10(dmatrix))


fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle("ideal distribution of 1s and 0s")
axs[0].imshow(dmatrix)
axs[0].set_title("original")
axs[1].imshow(np.log(dmatrix))
axs[1].set_title("log scale")

myblocks

# mysegments = [8, 19, 20, 21, 22, 13]
mysegments = [15, 29, 20, 21, 18]
np.cumsum(myblocks)
np.cumsum(mysegments)

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


# the blocks, explicitly indexed
articulate(myblocks)

temp = articulate(myblocks)

reduce(lambda x, y: x + list(y), temp, [])

# the original segments
mysegments
articulate(mysegments)

np.nancumsum(myblocks)
np.cumsum(mysegments)


# shuffle the order of the segments:
newOrder = np.random.permutation(len(mysegments))
newOrder

temp = articulate(mysegments)
temp

reindexlist = [temp[newOrder[i]] for i in newOrder]

reindexlist

reindexlist
# we shuffled the order, now lets reverse a few of the segments:
for i in [1, 4]:
    reindexlist[i] = np.flip(reindexlist[i])
reindexlist

reindexing = reduce(lambda x, y: x + list(y), reindexlist, [])
reindexing

# now lets see the original matrix and the matrix after the transformation:

newmatrix = np.zeros_like(mymatrix)
for i in range(len(mymatrix)):
    for j in range(len(mymatrix)):
        newmatrix[i, j] = mymatrix[reindexing[i], reindexing[j]]

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle("original block matrix anb its transformation")
axs[0].imshow(mymatrix)
# axs[0].set_title('')
axs[1].imshow(newmatrix)
# axs[1].set_title('')

# what we have to work with is newmatrix, as well as a list of the segments
# in their shuffled order, not the orignal of course.
newsegments = [mysegments[newOrder[i]] for i in range(len(newOrder))]
newsegments

# so we need to reshuffle the segments
newsegments
# so that eventually they will be order like that (after re-indexing)
mysegments
# and some of the new segments we'll have to reverse as well
# can we do that?


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
            x = i
            y = j
            if lreverse:
                x = iss[-1] - i
            if rreverse:
                y = jss[-1] - j
            # temp = np.exp(-np.abs(i-j))
            temp = np.exp(-np.abs(x - y))
            # we only care about interaction between the 2 segments and not
            # inside each one of them which wouldn't be affected by
            # rearrangement.
            s += refmat[iss[i], jss[j]] * temp
    return s


cs = articulate(newsegments)
cs = [list(x) for x in cs]
cs

xyz = np.zeros_like(newmatrix)
l = cs[0] + cs[1]
for i in l:
    for j in l:
        xyz[i, j] = newmatrix[i, j]
plt.imshow(xyz)

xyz = np.zeros_like(newmatrix)
l = cs[5]
for i in l:
    for j in l:
        xyz[i - l[0], j - l[0]] = newmatrix[i, j]
plt.imshow(xyz)

xyz = np.zeros_like(newmatrix)
l = cs[0] + cs[3]
# l = cs[0] + np.flip(cs[3])
for i in range(len(l)):
    for j in range(len(l)):
        xyz[i, j] = newmatrix[l[i], l[j]]
print(scorePair2(cs[0], cs[3], mymatrix))
print(scorePair2(cs[0], cs[3], newmatrix))
print(scorePair2(np.flip(cs[0]), cs[3], newmatrix))  # this is the problem?
plt.imshow(xyz)


plt.imshow(mymatrix)

for i in cs:
    for j in cs:
        # print(scorePair2(i,j, newmatrix))
        print(scorePair2(i, j, mymatrix))

reconstructionMatrix = np.zeros_like(newmatrix)

while len(cs) > 1:
    xs = cs.pop()
    print(xs)
    xsrev = xs.copy()
    xsrev.reverse()
    newmatrixrev = reindexMatrix(xs, xsrev, newmatrix)
    l = np.argmax([scorePair2(xs, y, newmatrix) for y in cs])
    sl = scorePair2(xs, cs[l], newmatrix)
    lrev = np.argmax(
        # [scorePair2(xsrev, y, newmatrix) for y in cs]
        [scorePair2(xs, y, newmatrixrev) for y in cs]
    )
    # slrev = scorePair2(xsrev, cs[l], newmatrix)
    slrev = scorePair2(xs, cs[lrev], newmatrixrev)
    r = np.argmax([scorePair2(y, xs, newmatrix) for y in cs])
    sr = scorePair2(cs[r], xs, newmatrix)
    rrev = np.argmax(
        # [scorePair2(y, xsrev, newmatrix) for y in cs]
        [scorePair2(y, xs, newmatrixrev) for y in cs]
    )
    # srrev = scorePair2(cs[r], xsrev,  newmatrix)
    srrev = scorePair2(cs[rrev], xs, newmatrixrev)
    iascores = [sl, slrev, sr, srrev]
    candidates = [cs[l], cs[lrev], cs[r], cs[rrev]]
    maxscore = np.max(iascores)
    if maxscore == sl:
        cs[l] = xs + cs[l]
    elif maxscore == sr:
        cs[r] = cs[r] + xs
    elif maxscore == lrev:
        cs[lrev] = xsrev + cs[lrev]
    else:
        cs[rrev] = cs[rrev] + xsrev


# reconstruction of the matrix
reconstructionMatrix = np.zeros_like(newmatrix)
myindices = cs[0]
myindices

n = len(newmatrix)
for i in range(n):
    for j in range(n):
        reconstructionMatrix[i, j] = newmatrix[myindices[i], myindices[j]]
        # reconstructionMatrix[myindices[i],myindices[j]] = newmatrix[i, j]

plt.imshow(newmatrix)

plt.imshow(reconstructionMatrix)

#### new try

reconstructionMatrix = np.zeros_like(newmatrix)
reconstructionMatrix = newmatrix.copy()

while len(cs) > 1:
    xs = cs.pop()
    print(xs)
    xsrev = xs.copy()
    xsrev.reverse()
    reconstructionMatrixrev = reindexMatrix(xs, xsrev, reconstructionMatrix)
    l = np.argmax([scorePair3(xs, y, reconstructionMatrix) for y in cs])
    sl = scorePair3(xs, cs[l], reconstructionMatrix)
    lrev = np.argmax(
        # [scorePair2(xsrev, y, reconstructionMatrix) for y in cs]
        # [scorePair2(xs, y, reconstructionMatrixrev) for y in cs]
        [scorePair3(xs, y, reconstructionMatrix, lreverse=True) for y in cs]
    )
    # slrev = scorePair2(xsrev, cs[l], reconstructionMatrix)
    slrev = scorePair3(xs, cs[lrev], reconstructionMatrix, lreverse=True)
    r = np.argmax([scorePair3(y, xs, reconstructionMatrix) for y in cs])
    sr = scorePair3(cs[r], xs, reconstructionMatrix)
    rrev = np.argmax(
        # [scorePair2(y, xsrev, reconstructionMatrix) for y in cs]
        [scorePair3(y, xs, reconstructionMatrix, rreverse=True) for y in cs]
    )
    # srrev = scorePair2(cs[r], xsrev,  reconstructionMatrix)
    srrev = scorePair3(cs[rrev], xs, reconstructionMatrix, rreverse=True)
    iascores = [sl, slrev, sr, srrev]
    candidates = [cs[l], cs[lrev], cs[r], cs[rrev]]
    maxscore = np.max(iascores)
    if maxscore == sl:
        cs[l] = xs + cs[l]
    elif maxscore == sr:
        cs[r] = cs[r] + xs
    elif maxscore == lrev:
        reconstructionMatrix = reindexMatrix(xs, xsrev, reconstructionMatrix)
        # cs[lrev] = xsrev + cs[lrev]
        cs[lrev] = xs + cs[lrev]
    else:
        reconstructionMatrix = reindexMatrix(xs, xsrev, reconstructionMatrix)
        # cs[rrev] = cs[rrev] + xsrev
        cs[rrev] = cs[rrev] + xs

n = len(newmatrix)
temp = np.zeros_like(newmatrix)
for i in range(n):
    for j in range(n):
        temp[i, j] = reconstructionMatrix[myindices[i], myindices[j]]
        # reconstructionMatrix[myindices[i],myindices[j]] = newmatrix[i, j]


######
plt.imshow(newmatrix)


cs = articulate(newsegments)
cs = [list(x) for x in cs]
cs
mysegments
newsegments


def improve(xss, yss, A):
    pass
