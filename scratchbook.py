import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

pd.read_csv(

T = np.identity(4)
T[0][3] = 1
T

T.sum(axis=0)

np.power(T, 3)

np.dot(T, T)

matrix_a = np.array([[2, 3, 4, 5], [4, 5, 6, 7]])
matrix_a

matrix_b = np.array([[12, 13, 14, 15], [14, 15, 16, 17]])

# add
print(np.add(matrix_a, matrix_b))
# subtract
print(np.subtract(matrix_a, matrix_b))
# multiply
print(np.multiply(matrix_a, matrix_b))
# divide
print(np.divide(matrix_a, matrix_b))

# sqrt
print(np.sqrt(matrix_a), np.sqrt(matrix_b))
# sum(x,axis)
print(np.sum(matrix_a, axis=0))
print(np.add(matrix_a, axis=1))
# Transpose
print(matrix_b.T)


def pwr(x, n):
    """raise to the n'th nonnegative
    integer power"""
    if n == 0:
        return np.identity(len(x))
    elif n == 1:
        return x
    else:
        return np.dot(x, pwr(x, n - 1))


def bpwr(x, n):
    if n == 0:
        return np.identity(len(x)).astype(x.dtype)
    elif n == 1:
        return (x != 0).astype(x.dtype)
    else:
        y = np.dot(x, pwr(x, n - 1))
        return (y != 0).astype(y.dtype)


T

pwr(T, 3)
bpwr(T, 3)

T = np.zeros((4, 4))
T
T[0] = [0, 1, 0, 0]
T[1] = [1, 1, 1, 0]
T[2] = [0, 0, 1, 0]
T[3] = [0, 0, 1, 1]
T

T.sum(axis=0)

T = T / T.sum(axis=0)
T

pwr(T, 0)
pwr(T, 1)
pwr(T, 2)
pwr(T, 3)
pwr(T, 4)
pwr(T, 5)
pwr(T, 6)

T

T.sum(axis=1)

#dasf
x = 1

df = pd.read_csv()

