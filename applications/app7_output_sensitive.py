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
X = noisy_circle(7, n_noise=0, perturb=0.05) ## 80, 30, 0.15 
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
S = sx.rips_complex(X, radius=0.75, p=2)
vid = sx.card(S,0)
S.insert([sx.Simplex([vid]) + sx.Simplex(s) for s in S])            ## cone the complex
coned_diam = cone_filter(sx.flag_filter(pdist(X)), vid = vid)       ## generate the coned filter 
coned_diam_filter = sx.fixed_filter(S, [coned_diam(s) for s in S])  ## optimize it for fast evaluation

## Create a new coned filtration
K = sx.RankFiltration(S, coned_diam_filter)
# K = sx.SetFiltration(S, coned_diam_filter)
index_filter = sx.fixed_filter(list(sx.faces(K)), np.arange(len(K)))
K.reindex(index_filter)
R, V = ph(K, output="RV")
dgms_index = generate_dgm(K, R, simplex_pairs=True)

## Plot the persistence diagram in the index-persistence plane
show(figure_dgm(dgms_index[1]))

## Print the filtration 
sx.print_complex(list(sx.faces(K)))

# %% Produce an array representing the persistence diagram
from pbsig.betti import betti_query
from pbsig.linalg import spectral_rank
K_simplices = list(map(sx.Simplex, sx.faces(K)))
M = len(K)
I = np.array([i for i,j in combinations(np.arange(M), 2)])
J = np.array([j for i,j in combinations(np.arange(M), 2)])
B = lambda i,j: list(betti_query(K_simplices, index_filter, spectral_rank, i = i, j = j, p = 1))[0]

from itertools import combinations
SR = sx.RankComplex(K_simplices) ## this doesn't work yet for unknown reasons
ST = sx.simplicial_complex(K_simplices, form='tree')

betti_test = np.array(list(betti_query(ST, index_filter, spectral_rank, i = I, j = J, p = 1, form='lo')))
betti_truth = np.array([np.sum(np.logical_and(dgms_index[1]['birth'] <= i, j < dgms_index[1]['death'])) for i,j in combinations(np.arange(M), 2)])
assert np.allclose(betti_truth - betti_test, 0)

# %% Debugging the performance
import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(betti_query)
profile.add_function(WeightedLaplacian.reweight)
profile.enable_by_count()
betti_test = np.array(list(betti_query(ST, index_filter, spectral_rank, i = I, j = J, p = 1, form='lo')))
profile.print_stats(output_unit=1e-3, stripzeros=True)

# For piazza post
# from scipy.sparse import coo_array
# ij = np.loadtxt("/Users/mpiekenbrock/Downloads/web-Google_10k.txt").astype(int)
# N = np.max(ij.max(axis=0))+1
# A = coo_array((np.repeat(True, len(ij)), (ij[:,0], ij[:,1])), dtype=bool, shape=(N, N))
# v = np.random.uniform(size = A.shape[1])
# timeit.timeit(lambda: A_csr @ v, number = 100)


#%%
np.array(list(betti_query(SR, index_filter, spectral_rank, i = I, j = J, p = 1)))

L = WeightedLaplacian(SR, p = 1)
L.reweight()


## %% 
_ = np.argmax(betti_truth - betti_test)
a, b = I[_], J[_]
betti_test[_], betti_truth[_]
B(a,b)
list(betti_query(K_simplices, index_filter, spectral_rank, i = a, j = b, p = 1, terms=True))

# %% Debugging third term
from pbsig.csgraph import WeightedLaplacian
from pbsig.linalg import PsdSolver
from pbsig.betti import smooth_dnstep, smooth_upstep
Lq = WeightedLaplacian(K, p = 1)
# fi_inc = smooth_dnstep(lb = a, ub = a+np.finfo(np.float64).eps)
# fi_exc = smooth_upstep(lb = ii, ub = ii+w)         
# fj_inc = smooth_dnstep(lb = jj-w, ub = jj+delta)

## This is reporting 11! So it's not right? where is the failure? 
D2 = sx.boundary_matrix(K, p = 2).tolil()
np.linalg.matrix_rank(D2[:,K.indices(p=2) <= b].todense())

Lq.reweight(fj_inc(sw), inc_all(fw)) # this one
t2 = matrix_func(q_solver(Lq.operator()))


# %% 
I0 = B(M // 2, M // 2 + 1)
I1 = B(0, M // 2)
I2 = I0 - I1

## The original formulation (seems correct)
ii = np.sum(K.indices(p=1) <= a)
jj = np.sum(K.indices(p=2) <= b)
D1 = sx.boundary_matrix(K, p=1).tolil()
D2 = sx.boundary_matrix(K, p=2).tolil()
A = D1[:,K.indices(p=1) <= a]
t0 = np.sum(K.indices(p=1) <= a)
t1 = np.linalg.matrix_rank(A.todense())
t23 = np.sum([piv < ii for piv in low_entry(R2[:, K.indices(p=1) <= b]) if piv != -1])
(t0, t1, t23)

## Direct approach using the full reduced R matrix (seems correct, yields t2 and t3)
D = sx.boundary_matrix(K).tolil()
R, V = ph(K, output="RV")
for cc, (i,s) in enumerate(K):
  if sx.dim(s) != 2:
    R[:,cc] = 0
R.eliminate_zeros()
# np.linalg.matrix_rank(R.todense()) # this reports 9
t2 = np.linalg.matrix_rank(R[:,K.indices() <= b].todense()) ## this one is wrong in code
t3 = np.linalg.matrix_rank(R[:,K.indices() <= b][K.indices() > a,:].todense())
# t23 = np.sum(np.logical_and(np.arange(len(K)) <= b, np.logical_and(0 <= low_entry(R), low_entry(R)  <= a)))

## Extract D2 
nz_col_ind = np.flatnonzero(np.abs(R).sum(axis=0) != 0)
nz_row_ind = np.flatnonzero(np.abs(R).sum(axis=1) != 0)
R.todense()[np.ix_(nz_row_ind, nz_col_ind)]

edge_ind = np.array(list(K.indices(p=1)))
tri_ind = np.array(list(K.indices(p=2)))
edge_tri_ind = np.ix_(edge_ind.astype(int), tri_ind.astype(int))
R2 = R.todense()[edge_tri_ind].astype(int)
D2 = sx.boundary_matrix(K, p = 2).todense()
V2 = V.todense()[np.ix_(tri_ind, tri_ind)]
assert D2.shape == R2.shape


## AHH the full boundary matrix ordering doesn't match the p-specialized one
## Try use set complex in the above 
## D2 is incorrect! Full boundary matrix is correct!
## The right one always assigns (+1, -1, +1) to (3,4,5) => (3,4), (3,5), (4,5) in lex order
sx.boundary_matrix(K)[edge_tri_ind].todense() - D2
wrong = D2[:,0]
right = sx.boundary_matrix(K)[edge_tri_ind].todense()[:,0]



## Cancellations indeed happen
(D @ V).todense()[edge_tri_ind].astype(int)

R2[:,:8]
D2_copy = D2[:,:8].copy()

np.linalg.matrix_rank(D2[:,:8][-2:])

gen = V.todense()[np.ix_(K.indices(p=2).astype(int),  K.indices(p=2).astype(int))][:,7]
gen = np.ravel(gen)
-D2[:,0] + D2[:,5] - D2[:,6] + D2[:,7]


## The non-zero status of the reduced and boundary matrices is different!
## Update: this seems to be because reduction was done mod2, which can differ from real-value coefficients!
np.linalg.svd(R2[:,:8].astype(np.float32))[1]
np.linalg.svd(D2[:,:8].astype(np.float32))[1]

## This says that the fourth term must be all zero 
D2[:, K.indices(p=2) <= b][K.indices(p=1) >= a,:].todense()

## This says the 3rd + 4rth is 14! 
list(betti_query(K_simplices, index_filter, spectral_rank, i = M // 2, j = M // 2, p = 1, terms=True))[0]

## Exact 
np.sum(np.logical_and(dgms_index[1]['birth'] <= a, dgms_index[1]['death'] > b))

from pbsig.persistence import persistent_betti, reduction_pHcol, low_entry

## Back to the basics: says 3rd + 4rth is 14! 
D1 = sx.boundary_matrix(K, p=1).tolil()
D2 = sx.boundary_matrix(K, p=2).tolil()
# ii = len([s for s in sx.faces(K,1) if s.value <= a]) ## should be 12 
# jj = len([s for s in sx.faces(K,2) if s.value <= b]) ## should be 14
ii = np.sum(K.indices(p=1) <= a)
jj = np.sum(K.indices(p=2) <= b)
persistent_betti(D1, D2, ii, jj, summands=True) ## this reports the same 

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