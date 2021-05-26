#%%
import numpy as np
import toolz

d = dict(zip("acgt", range(4)))
d
def enc(s):
    if s=="":
        return 0
    else:
        return 4*enc(s[0:-1]) + d[s[-1]]

h1 = lambda x: enc(x)*19 % 39
h2 = lambda x: enc(x)*43 % 39
#%%1

enc("")
enc("a")
enc("acgttt")

h1("acgttt")


bf1 = np.zeros(39)
bf2 = np.zeros(39)


S1  = ["acgtagc", "ctaga"]
S2 = ["tacgttgc"]

def insertOneKmer(v, kmer, h):
    """v: bitvector
    kmer: kmer (string)
    h: hash function
    """
    v[h(kmer)] = 1

def insertKmers(v, k, seq, hs):
    """v: bitvector
    k: kmer size,
    seq: sequence to hash kmers from
    hs: list of hash functions
    """
    n = len(seq)
    for i in range(n-k+1):
        kmer = seq[1:i+k]
        for h in hs:
            insertOneKmer(v, kmer, h)


for s in S1:
    insertKmes(bf1, 3, s, [h1, h2])

insertKmers(bf2, 3, S2[0], [h1, h1])


