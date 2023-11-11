# %%
import numpy as np
import splex as sx
from typing import * 
from scipy.sparse import spmatrix
from scipy.spatial.distance import pdist
from numpy.typing import ArrayLike
from bokeh.io import output_notebook, export_png
from bokeh.plotting import figure, show
from bokeh.io import export_png
from pbsig.utility import progressbar
from more_itertools import chunked
from pbsig.vineyards import linear_homotopy, transpose_rv
from math import comb
from pbsig.utility import progressbar
from pbsig.vis import bin_color
from pbsig.vis import figure_complex
output_notebook()

# %%
from pbsig.datasets import noisy_circle
from splex.geometry import delaunay_complex
np.random.seed(1234)
X = noisy_circle(10, n_noise=0, perturb=0.05) ## 80, 30, 0.15 
S = sx.rips_complex(X, radius=0.75, p=2)

# %% Peek at the complex 
show(figure_complex(S, X))

#%% Get the rips diagram 
from pbsig.vis import figure_dgm
from pbsig.persistence import ph, generate_dgm
K = sx.rips_filtration(X, radius=0.75, p=2) ## need triangles to collapse 
R, V = ph(K, output="RV")
dgms = generate_dgm(K, R, simplex_pairs=True)

## Plot the persistence diagram in the filter plane
show(figure_dgm(dgms[1]))

# %% First, re-index to the index filtration
K.reindex(np.arange(len(K)))
dgms_index = generate_dgm(K, R, simplex_pairs=True)

## Plot the persistence diagram in the index-persistence plane
show(figure_dgm(dgms_index[1]))

# %% 
from pbsig.betti import betti_query
from pbsig.linalg import spectral_rank
K_simplices = list(map(sx.Simplex, sx.faces(K)))
filter_f = sx.fixed_filter(K_simplices, np.arange(len(K)))

B = lambda i,j: list(betti_query(K_simplices, filter_f, spectral_rank, i = i, j = j, p = 1))[0]
M = len(K)
I0 = B(M // 2, M // 2)
I1 = B(0, M // 2)
I2 = I0 - I1

from pbsig.persistence import persistent_betti, reduction_pHcol, low_entry

a, b = 20.0, 40.0
D1 = sx.boundary_matrix(K, p=1).tolil()
D2 = sx.boundary_matrix(K, p=2).tolil()
ii = len([s for s in sx.faces(K,1) if s.value <= i]) ## should be 12 
jj = len([s for s in sx.faces(K,2) if s.value <= j]) ## should be 14
persistent_betti(D1, D2, ii, jj, summands=False)

## Manual calculation using Chen
R1, R2, V1, V2 = reduction_pHcol(D1, D2)
t0 = ii
t1 = np.sum(low_entry(R1[:,:ii]) != -1)
t2 = np.sum(low_entry(R2[:,:jj]) != -1)
t3 = np.sum(low_entry(R2[ii:,:jj]) != -1)
(t0,t1,t2,t3)
t0 - t1 - t2 + t3
## Indeed, the terms are right, but the face poset isnt respected!

  # t2 = 0 if np.prod(D1[:,:i].shape) == 0 else np.linalg.matrix_rank(D1[:,:i].A)
  # D2_tmp = D2[:,:j]
  # t3_1 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
  # D2_tmp = D2_tmp[i:,:] # note i is 1-based, so this is like (i+1)
  # t3_2 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
# t0,t1,t2,t3


B(M // 2, M)



filter_f(sx.faces(K,2))
{ s : k for k,s in K }