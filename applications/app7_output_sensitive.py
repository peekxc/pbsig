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
from itertools import combinations
K_simplices = list(map(sx.Simplex, sx.faces(K)))
M = len(K)
I = np.array([i for i,j in combinations(np.arange(M), 2)])
J = np.array([j for i,j in combinations(np.arange(M), 2)])
B = lambda i,j: list(betti_query(K_simplices, index_filter, spectral_rank, i = i, j = j, p = 1))[0]

from itertools import combinations
SR = sx.RankComplex(K_simplices) ## this doesn't work yet for unknown reasons
ST = sx.simplicial_complex(K_simplices, form='tree')

# betti_test = np.array(list(betti_query(ST, index_filter, p=1, matrix_func=spectral_rank, i = I, j = J, form='lo')))
# betti_truth = np.array([np.sum(np.logical_and(dgms_index[1]['birth'] <= i, j < dgms_index[1]['death'])) for i,j in combinations(np.arange(M), 2)])
# assert np.allclose(betti_truth - betti_test, 0)

# %% Assemble the diagram in the index-persistence plane, using top-down approach 
from pbsig.betti import BettiQuery

## View the filtration
K.print()
p = figure_dgm(dgms_index[1])
show(p)
print(dgms_index[1])

## Setup the query model
m = len(K)
query = BettiQuery(ST, p = 1)

a,b,c,d = 0, m // 2, m // 2 + 1, m

p.rect(x=a + (b-a) / 2, y=c + (d-c) / 2, width=b-a, height=d-c, fill_color=None)
show(p)

## Create a simplex to value map 
s_map = { sx.Simplex(s) : i for i, s in K }
query.weights[0] = np.array([s_map[s] for s in sx.faces(ST, 0)])
query.weights[1] = np.array([s_map[s] for s in sx.faces(ST, 1)])
query.weights[2] = np.array([s_map[s] for s in sx.faces(ST, 2)])

## Replace the solver 
from primate.functional import numrank
def my_solver(L):
  if len(L.data) == 0: 
    return 0 
  elif L.shape[0] <= 64:
    return np.linalg.matrix_rank(L.todense())
  else: 
    return numrank(L, atol=0.50, gap="simple")

query.q_solver = my_solver
query.p_solver = my_solver

def generate_boxes(a: int, b: int, res: dict = {}, index: int = 0):
  r = (a, (a+b) // 2, (a+b) // 2 + 1, b)
  res[index] = r
  if abs(a-b) <= 1: 
    return  
  else:
    generate_boxes(a, (a+b) // 2, res, 2*index + 1)
    generate_boxes((a+b) // 2, b, res, 2*index + 2)

boxes = {}
generate_boxes(0, m, boxes)

birth = dgms_index[1]['birth']
death = dgms_index[1]['death']

def dumb_query(i,j,k,l):
  valid_births = np.logical_and(i <= birth, birth <= j) 
  valid_deaths = np.logical_and(k <= death, death <= l) 
  return np.sum(np.logical_and(valid_births, valid_deaths))

## Find creators in range [a, mid] whose destroyers lie in [mid+1, b]
midpoint = lambda a,b: (a + b) // 2

a, b = 0, len(K)
mid = a + b // 2
# show(p)
dumb_query(a, midpoint(a,b), midpoint(a,b)+1, b)

def bisection_tree(i1, i2, j1, j2, mu: int, query_fun: Callable):
  print(f"({i1},{i2},{j1},{j2}) = {mu}")
  if mu == 0: 
    return
  elif i1 == i2 and mu == 1:
    print(f"Creator found at index {i1} with destroyer in [{j1}, {j2}]") 
  else:
    mid = midpoint(i1, i2)
    mu_L = query_fun(i1, mid, j1, j2)
    mu_R = mu - mu_L 
    assert mu_R == query_fun(mid+1, i2, j1, j2)
    bisection_tree(i1, mid, j1, j2, mu_L, query_fun)
    bisection_tree(mid+1, i2, j1, j2, mu_R, query_fun)

a, b = 0, len(K)
mid = a + b // 2
mu_init = dumb_query(a, midpoint(a,b), midpoint(a,b)+1, b)
bisection_tree(a, midpoint(a,b), midpoint(a,b)+1, b, mu_init, dumb_query)

assert dumb_query(a, mid, mid+1, b) == (dumb_query(0, 9, mid+1, b) + dumb_query(9, 18, mid+1, b))

## Now, for EACH creator found in bisection, we find unique j by \in [j1,j2] binary search
def find_destroyer(creator: int, l: int, r: int, query_fun: Callable):
  ii = creator 
  mu_j = dumb_query(ii, ii, l, r)
  if mu_j == 0: 
    return ii
  while l != r:
    # print(f"l, r: ({l}, {r})")
    mu_L = dumb_query(ii, ii, l, midpoint(l,r))
    if mu_L != 0:
      r = (l + r) // 2      ## in the left portion 
    else: 
      l = (l + r) // 2 + 1  ## in the right partition
  return l 

## B0: First creator/destroyer pair
a, b = 0, len(K)
mu_init = dumb_query(0, midpoint(a,b), midpoint(a,b)+1, b)
bisection_tree(a, midpoint(a,b), midpoint(a,b)+1, b, mu_init, dumb_query)
find_destroyer(14, mid+1, b, dumb_query) # 33 

## B1: Second pair going down
mid = midpoint(a,b)
mu_init = dumb_query(a, midpoint(a,mid), midpoint(a,mid)+1, mid)
bisection_tree(a, midpoint(a,mid), midpoint(a,mid)+1, mid, mu_init, dumb_query)

## B1: Second pair going right
mid = midpoint(a,b)
mu_init = dumb_query(mid+1, midpoint(mid+1,b), midpoint(mid+1,b)+1, b)
bisection_tree(mid+1, midpoint(mid+1,b), midpoint(mid+1,b)+1, b, mu_init, dumb_query)
find_destroyer(23, midpoint(mid+1,b)+1, b, dumb_query) # 29
find_destroyer(24, midpoint(mid+1,b)+1, b, dumb_query) # 30
find_destroyer(25, midpoint(mid+1,b)+1, b, dumb_query) # 32

## TODO: 
# 1. Write Points-in-box function (bisection tree + destroyer)
# 2. Write code to enumerate all B* boxes in order of area + left-to-right
# 3. Compute all pairs 
# 4. Determine position needed to compute all pairs up to gamma-persistence
# 5. Biject it all back to the function domain
# 6. INterface it up 
# 7. Test test test
dgms_index[1]

# from pbsig.persistence import bisection_tree_top_down



# list(bisection_tree_top_down(dumb_query, ind = (a, mid, mid+1, b), mu=1, splitrows=False, verbose=True))


mu0 = dumb_query(0, mid, mid + 1, b)

# mu0 = query(a, mid, mid + 1, b)


mu_left = dumb_query(l, mid, mid + 1, r)

l, mid, r = mid + 1, mid + b // 2, b
mu_right = query(l, mid, mid + 1, r)
mu_right



## Recurse 
query(a,b,c,d)

## root 
a,b,c,d = 0, m // 2, m // 2, m

## left 
a,b,c,d = a, a + (b - a) // 2, a + (b - a) // 2, b

## right 
a,b,c,d = a, a + (b - a) // 2, a + (b - a) // 2, b


query(a,b,c,d, terms=False)
numrank(query.Lq.operator(deflate=True), atol = 0.50)

A = query.operator(1, a,b,c,d)
numrank(A)

query.weights


D1 = sx.boundary_matrix(K, p = 1)
np.array([i for i,s in K if sx.dim(s) == 2])




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
from pbsig.linalg import WeightedLaplacian
from pbsig.betti import mu_query

L = WeightedLaplacian(SR, p = 1)








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