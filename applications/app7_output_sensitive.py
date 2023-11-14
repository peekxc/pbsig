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
X = noisy_circle(5, n_noise=0, perturb=0.05) ## 80, 30, 0.15 
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

# %% Capture the creators 
# 1. First, cone the complex + the filter
# 2. Second, re-index to the index filtration
from pbsig.betti import cone_filter
vid = sx.card(S,0)
S.insert([sx.Simplex([vid]) + sx.Simplex(s) for s in S])            ## cone the complex
coned_diam = cone_filter(sx.flag_filter(pdist(X)), vid = vid)       ## generate the coned filter 
coned_diam_filter = sx.fixed_filter(S, [coned_diam(s) for s in S])  ## optimize it for fast evaluation

## Create a new coned filtration
K = sx.RankFiltration(S, coned_diam_filter)
index_filter = sx.fixed_filter(list(sx.faces(K)), np.arange(len(K)))
K.reindex(index_filter)
R, V = ph(K, output="RV")
dgms_index = generate_dgm(K, R, simplex_pairs=True)

## Plot the persistence diagram in the index-persistence plane
show(figure_dgm(dgms_index[1]))


# %% Produce an array representing the persistence diagram
from pbsig.betti import betti_query
from pbsig.linalg import spectral_rank
K_simplices = list(map(sx.Simplex, sx.faces(K)))

B = lambda i,j: list(betti_query(K_simplices, index_filter, spectral_rank, i = i, j = j, p = 1))[0]
M = len(K)
I0 = B(M // 2, M // 2 + 1)
I1 = B(0, M // 2)
I2 = I0 - I1



## The original formulation
D1 = sx.boundary_matrix(K, p=1).tolil()
D2 = sx.boundary_matrix(K, p=2).tolil()
t0 = np.max(D1[:,K.indices(p=1) <= a].shape)
t1 = np.linalg.matrix_rank(D1[:,K.indices(p=1) <= a].todense())

## This says the 3rd + 4rth is 17! 
t23 = np.sum([0 <= piv < ii for piv in low_entry(R2[:, K.indices(p=1) <= b])])

## Let's try with the full boundary matrix
D = sx.boundary_matrix(K).tolil()
R, V = ph(K, output="RV")

## This says 3rd + 4rth is 12! which seems more correct 
for cc, (i,s) in enumerate(K):
  if sx.dim(s) != 2:
    R[:,cc] = 0
R.eliminate_zeros()
np.sum(np.logical_and(np.arange(len(K)) <= b, np.logical_and(0 <= low_entry(R), low_entry(R)  <= a)))

## This says that the fourth term must be all zero 
D2[:, K.indices(p=2) <= b][K.indices(p=1) >= a,:].todense()

## This says the 3rd + 4rth is 14! 
list(betti_query(K_simplices, index_filter, spectral_rank, i = M // 2, j = M // 2, p = 1, terms=True))[0]

## Exact 
np.sum(np.logical_and(dgms_index[1]['birth'] <= a, dgms_index[1]['death'] > b))

from pbsig.persistence import persistent_betti, reduction_pHcol, low_entry

## Back to the basics: says 3rd + 4rth is 14! 
a, b = M // 2, (M // 2) + 1
D1 = sx.boundary_matrix(K, p=1).tolil()
D2 = sx.boundary_matrix(K, p=2).tolil()
# ii = len([s for s in sx.faces(K,1) if s.value <= a]) ## should be 12 
# jj = len([s for s in sx.faces(K,2) if s.value <= b]) ## should be 14
ii = np.sum(K.indices(p=1) <= a)
jj = np.sum(K.indices(p=2) <= b)
persistent_betti(D1, D2, ii, jj, summands=True)

sx.faces(K,1)[:ii,:]
sx.faces(K,2)[:jj,:]

## Check global face poset 
for i, s in K: assert all([K.index(face) <= i for face in sx.faces(s)])

## Check local face poset wrt (ii,jj)
S_test = sx.simplicial_complex(sx.faces(K,1)[:ii,:])
assert np.all([np.all([f in S_test for f in sx.faces(s,1)]) for s in sx.faces(K,2)[:jj,:]])


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