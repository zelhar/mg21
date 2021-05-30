import straw
import numpy as np
from scipy.sparse import coo_matrix

#https://colab.research.google.com/drive/1548GgZe7ndeZseaIQ1YQxnB5rMZWSsSj

straw.straw?

spmat = straw.straw(
    "KR",
    "../../mnt/Yiftach_Kolb_project_hic_genome_reconstruction/191-98_hg19_no_hap_EBV_MAPQ30_merged.hic",
    "1", "2",
    unit="BP",
    binsize=500000,
)



