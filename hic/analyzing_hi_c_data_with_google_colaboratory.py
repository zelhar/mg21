# -*- coding: utf-8 -*-
"""Analyzing Hi-C Data with Google Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1548GgZe7ndeZseaIQ1YQxnB5rMZWSsSj

<h1>Use Straw to Analyze Hi-C Data!</h1>

# Where are we?

The document you are reading is a  [Jupyter notebook](https://jupyter.org/), hosted in Google Colaboratory. It is not a static page, but an interactive environment that lets you write and execute code in Python and other languages.

For example, here is a **code cell** with a short Python script that computes a value, stores it in a variable, and prints the result:
"""

seconds_in_a_day = 24 * 60 * 60
print(seconds_in_a_day)

"""To execute the code in the above cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut "Command/Ctrl+Enter".

All cells modify the same global state, so variables that you define by executing a cell can be used in other cells:
"""

seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week

"""For more information about working with Colaboratory notebooks, see [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb).

## More Resources

Learn how to make the most of Python, Jupyter, Colaboratory, and related tools with these resources:

### Working with Notebooks in Colaboratory
- [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb)
- [Uploading and Downloading Files](/notebooks/io.ipynb)
- [Guide to Markdown](/notebooks/markdown_guide.ipynb)
- [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
- [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
- [Interactive forms](/notebooks/forms.ipynb)
- [Interactive widgets](/notebooks/widgets.ipynb)

# Using Straw

## Streaming in Hi-C Data

The ENCODE suite of tools for accessing and analyzing Hi-C Data includes the Straw API for streaming Hi-C data without needing to download any files. Straw currently supports Python, C/C++, R, MATLAB, and JavaScript. Here, we will be using the Straw API for Python.

The first step is to install straw on the machine:
"""

!pip install hic-straw

"""Now we import the straw library (as well as some additional common libraries) into our code:"""

import straw
import numpy as np
from scipy.sparse import coo_matrix

"""Here's a sample call which will pull data from the Hi-C file. This call extracts all reads from chromosome 4 at 500KB resolution with KR (Knight-Ruiz balancing algorithm) normalization from the combined MAPQ 30 map from Rao and Huntley et al. 2014"""

result = straw.straw('KR', 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined_30.hic', '4', '4', 'BP', 500000)

# printing the first 10 rows from the sparse format
for i in range(10):
   print("{0}\t{1}\t{2}".format(result[0][i], result[1][i], result[2][i]))

"""Here's a small helper function to extract data from **along the diagonal** of the Hi-C map and put it into a dense format (with symmetry)."""

def extract_data_along_hic_diagonal(url, resolution, coordinates):
  # assumes KR normalization and BP resolutions
  result = straw.straw("KR", url, coordinates, coordinates, "BP", resolution)
  
  # convert genomic position to a relative bin position
  row_indices = np.asarray(result[0])/resolution
  col_indices = np.asarray(result[1])/resolution
  row_indices = row_indices - np.min(row_indices)
  col_indices = col_indices - np.min(col_indices)
  
  # put into a sparse matrix format
  data = result[2]
  max_size = int(np.max([np.max(row_indices), np.max(col_indices)]))+1
  mat_coo = coo_matrix((data, (row_indices.astype(int), col_indices.astype(int))), shape=(max_size, max_size))
  
  # put data into a dense matrix format
  dense = mat_coo.toarray()
  dense_full = dense.T + dense # make matrix symmetric
  np.fill_diagonal(dense_full, np.diag(dense)) # prevent doubling of diagonal from prior step
  dense_full[np.isnan(dense_full)] = 0 # set NaNs to zero
  return dense_full

"""## Plotting Data
Now let's plot some data!
"""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt 
import seaborn as sns


def plot_hic_map(dense_matrix, maxcolor):
    plt.matshow(dense_matrix, cmap=LinearSegmentedColormap.from_list("bright_red",[(1,1,1),(1,0,0)]), vmin=0, vmax=maxcolor)
    plt.show()


data = extract_data_along_hic_diagonal('https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/primary.hic', 50000, '1:2000000:10000000')
plot_hic_map(data, 200)

"""## Analyzing Data
Now let's do some simple correlation analysis.
"""

import numpy as np
from scipy import sparse
from scipy import stats

res=250000

# effectively a local chrom.sizes file for hg19
#chrs=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','Y')
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
        "Y": 59373566
} 

# use only the first 3 chromosomes for now
chrs=(1,2,3)

def getMatrixAsFlattenedVector(normalization, filepath, chrom, resolution, dozscore=False):
  for i in chrs:
    result = straw.straw(normalization, filepath, chrom, chrom, 'BP', resolution)
    I=np.array(result[0][:])/res
    J=np.array(result[1][:])/res
    V=np.array(result[2][:])
    sz=int(chr_sizes[str(i)]/res)+1
    M=sparse.coo_matrix((V,(I,J)),shape=(sz,sz)).tocsr()
    # make symmetric instead of upper triangular
    N=M+M.T-sparse.diags(M.diagonal(),dtype=int)
    A=N.reshape(1,sz*sz)
    if (i is not 1):
      vector = np.concatenate([vector, A.toarray().flatten()])
    else:
      vector = A.toarray().flatten()
    if dozscore:
      vector = stats.zscore(vector)
    return vector
          
vector1 = getMatrixAsFlattenedVector('KR', 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/primary.hic', str(i), res)
vector2 = getMatrixAsFlattenedVector('KR', 'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/replicate.hic', str(i), res)

print('Pearson correlation coefficient and p-value for testing non-correlation')
print(stats.pearsonr(vector1, vector2))

print('Spearman rank-order correlation coefficient and p-value to test for non-correlation')
print(stats.spearmanr(vector1, vector2))