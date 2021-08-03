# Place here the functions I am using
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import toolz
from toolz.curried import reduce
from toolz.curried import concatv, concat
from scipy.linalg import block_diag, tri, tril, triu
from matplotlib import cm
from numpy import flip
from numpy import log
from scipy.signal import convolve2d


def blockMatrix(blocks):
    """creates a bloack 0-1 matrix.
    param blocks: list of non-negative integers which is the size of the blocks.
    a 0 block size corresponds to a 0 on the main diagonal.
    """
    blocks = np.array(blocks).astype("int64")

    def f(x):
        return 1 if x == 0 else x

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


def translocationPermutation(i, j, k, n):
    """
    Swap [i,..,j) with [j,....k)
    Returns a permutation on [0....n], which represents
    the translocation permutation.
    It should be i<j<k<n, otherwise they will be sorted to be in that order.
    """
    i, j, k, n = sorted((i, j, k, n))  # should be 0 <= i<j<k<=n
    l = range(0, i)
    a = range(i, j)
    b = range(j, k)
    r = range(k, n)
    P = concatv(l, b, a, r)
    return list(P)


def inversionPermutation(i, j, n):
    """
    Inverts the range [i,j) within [1,n).
    Parmas should be 0<=i<j<=n.
    """
    i, j, n = sorted((i, j, n))  # should be 0 <= i<j<k<=n
    l = range(0, i)
    a = reversed(range(i, j))
    r = range(j, n)
    P = concatv(l, a, r)
    return list(P)


def rangeSwapPermutation(i, j, k, l, n):
    """
    swaps the index range [i,j) with [k,l)
    0<=i<j<k<l<=n
    """
    i, j, k, l, n = sorted((i, j, k, l, n))
    a = range(0, i)
    b = range(i, j)
    c = range(j, k)
    d = range(k, l)
    e = range(l, n)
    p = concatv(a, d, c, b, e)
    return list(p)


def permutationMatrix(ls):
    """returns a permutation matrix of size len(ls)^2.
    param ls: should be a reordering of range(len(l s)), which defines the
    permutation on the ROWS.
    returns a permutation matrix P.
    np.dot(P,A) should be rearrangement of the rows of A according to P.
    To permute the columns of a matrix A use:
    Q = np.transpose(P),
     then: np.dot(A,Q).
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


def scoreMatrix(n, logscore=False):
    """The score function of the matrix. The assumption is that the true
    arrangement maximized the interaction close to the main diagonal.
    The total sum of the interaction is an invariant, preserved by permuations.
    param n: size of ca 2-d n over n array.
    returns the score matrix, which is used to calculate the score of any given
    n^2 matrix.
    if logscore==True it returns the log2 of the score matrix.
    """
    s = np.arange(n)
    if logscore:
        s = -s
    else:
        s = np.exp(-s)
    S = np.zeros((n, n))
    for i in range(n):
        S[i][i:] = s[: n - i]
    S += triu(S, 1).T
    return S


def score(A, S=None):
    """returns the weighted sum (by the score matrix S) of
    the matrix A
    """
    if S == None:
        S = scoreMatrix(len(A))
    return np.sum(A * S)


def constrainMatrix(ls, A, rs=None):
    """Returns a matrix of the same dimension as A, but every entry with
    an index (either row or column) not in ls is 0.
    """
    B = np.zeros_like(A)
    if rs == None:
        B[np.ix_(ls, ls)] = A[np.ix_(ls, ls)]
        return B
    else:
        B[np.ix_(ls, rs)] = A[np.ix_(ls, rs)]
        return B


def resetIndices(ls, A, rs=None):
    """essentially returns the constraint of A to the complement indices of ls,
    by reseting all the indices in ls to 0.
    """
    if rs == None:
        rs = ls
    B = A.copy()
    B[ls, :] = 0
    B[:, rs] = 0
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


def interactionScore(I, J, T, logscore=False):
    """
    for disjoint subsets of indices, I,J of
    a matrix T, we define their interaction score as:
        ias(I,J,T) = \sum_{i \in I,j \in J} T[i,j]exp(-|i-j|)
    so basically it is the sum of all T[i,j] elements
    penalized by the exponential of the pair-wise distance.
    """
    n = len(T)
    S = scoreMatrix(n, logscore=True)
    C = constrainMatrix(I, T, J)
    # print(C * S)
    ia = np.sum(C * S)
    return ia


def alternativeScore(I, J, T, logscore=False):
    """The score of I,J if there were the neighboring ranges
    [0,|I|) and [|I|, |I|+|J|)"""
    n = len(T)
    iss = [i for i in range(len(I))]
    jss = [j for j in range(len(J))]
    C = np.zeros_like(T)
    C[np.ix_(iss, jss)] = T[np.ix_(I, J)]
    S = scoreMatrix(n, logscore=True)
    print(C * S)
    ia = np.sum(C * S)
    return ia


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
                y = jss[-1 - j]
            # temp = np.exp(-np.abs(i-j))
            # temp = np.exp(-np.abs(x - y))
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


def plotMat(x, y):
    n = len(x)
    labels = list(map(str, range(n)))
    figure = plt.figure()
    axes = figure.add_subplot(121)
    axes2 = figure.add_subplot(122)
    caxes = axes.matshow(x, interpolation="nearest")
    caxes2 = axes2.matshow(y, interpolation="nearest")
    # figure.colorbar(caxes)
    axes.set_xticks(range(n), minor=True)
    axes.set_yticks(range(n), minor=True)
    axes2.set_xticks(range(n), minor=True)
    axes2.set_yticks(range(n), minor=True)
    # axes.set_xticklabels(labels, minor=False)
    # axes.set_yticklabels(labels, minor=False)
    # axes = figure.add_subplot(111)


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


def indexing(arts):
    return reduce(lambda x, y: x + list(y), arts, [])


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
    sl = scorePair3(iss, jss, A)
    slrv = scorePair3(iss, jss, A, lreverse=True)
    sr = scorePair3(jss, iss, A)
    srrv = scorePair3(jss, iss, A, rreverse=True)
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
        _, B = swap2(1, 0, A, xs)
        return mysegs, B
    elif t == slrv:
        # first flip the segment 0
        B = flip1(0, A, xs)
        return mysegs, B
    else:
        # first flip the segment 0
        B = flip1(0, A, xs)
        # then make the switch
        _, B = swap2(1, 0, B, xs)
        return mysegs, B
