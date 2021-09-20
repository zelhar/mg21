import numpy as np

if __name__ == '__main__':
    from mymodule import *
else:
    from .mymodule import *


A = np.random.rand(10,8) * 1e-5

plt.matshow(A)


