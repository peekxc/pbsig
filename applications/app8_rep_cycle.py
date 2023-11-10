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

# %% Load data 
from pbsig.datasets import noisy_circle
from splex.geometry import delaunay_complex
np.random.seed(1234)
X = noisy_circle(10, n_noise=0, perturb=0.05) ## 80, 30, 0.15 
S = sx.rips_complex(X, radius=0.75, p=1)

# %% Peek at the complex 
show(figure_complex(S, X))

#%% Get the rips diagram 
from pbsig.persistence import ph, generate_dgm
K = sx.rips_filtration(X, radius=0.75, p=2) ## need triangles to collapse 
R, V = ph(K, output="RV")
dgms = generate_dgm(K, R, simplex_pairs=True)

## Get the essential creator simplex + its index in the filtration
assert len(dgms[1]) == 1
ess_creator = np.take(dgms[1]['creators'], 0)
ess_index = np.take(np.flatnonzero([sx.Simplex(s[1]) == sx.Simplex(ess_creator) for s in K]), 0)

## Verify it's column is 0
assert len(R[:,ess_index].data) == 0

## It's generator must be stored in V
cycle_indices = V[:,ess_index].indices
cycle_simplices = [sx.Simplex(K[i]) for i in cycle_indices]

# %% Plot the generator in question 
p = figure_complex(S, X)
p.multi_line(
  xs = [X[[i,j],0] for i,j in cycle_simplices], 
  ys = [X[[i,j],1] for i,j in cycle_simplices], 
  line_color = 'blue', line_width = 3
)
p.line(x=X[ess_creator, 0], y=X[ess_creator, 1], line_color='red', line_width=3)
show(p)

# %% Obtain a representative cycle with the filtered boundary matrix
from scipy.sparse.linalg import lsqr
D = sx.boundary_matrix(K)                 ## Filtered boundary matrix
A = D[:ess_index, :ess_index]             ## All simplices preceeding creator
b = D[:ess_index, [ess_index]].todense()  ## Boundary chain of creator simplex

## Since A is singular, it cannot be factored directly, so solve least-squares min ||Ax - b||_2
x, istop, itn, r1norm = lsqr(A.tocsc(), b)[:4]
cycle_rep_indices = np.flatnonzero(x != 0)
rep_cycle_simplices = [sx.Simplex(K[i]) for i in cycle_rep_indices]

## Plot the representative cycle
p = figure_complex(S, X)
p.multi_line(
  xs = [X[[i,j],0] for i,j in rep_cycle_simplices], 
  ys = [X[[i,j],1] for i,j in rep_cycle_simplices], 
  line_color = 'blue', line_width = 3
)
p.line(x=X[ess_creator, 0], y=X[ess_creator, 1], line_color='red', line_width=3)
show(p)

# %% Replicate the results but now with the permutation-invariant approach

## Start with the full boundary matrix in the lex order
D = sx.boundary_matrix(S).tocoo()

## Get the indices of the creator and its predecessors
diam_filter = sx.flag_filter(pdist(X))
ess_index = np.take(np.flatnonzero([sx.Simplex(s) == sx.Simplex(ess_creator) for s in S]), 0)
pred_ind = np.setdiff1d(np.flatnonzero(diam_filter(sx.faces(S)) <= diam_filter(ess_creator)), ess_index)

## Zero out the creator + its successors columns
b = D.tocsc()[:,[ess_index]].todense()
nz_entries = pred_ind[np.searchsorted(pred_ind, D.col)] == D.col
D.data[~nz_entries] = 0.0
D.eliminate_zeros()

## Solve the linear system 
x, istop, itn, r1norm = lsqr(D.tocsc(), b)[:4]

## Get the simplices fo the rep. cycle 
rep_cycle_simplices = [s for s, a in zip(S, x) if a != 0]

## Plot the representative cycle
p = figure_complex(S, X)
p.multi_line(
  xs = [X[[i,j],0] for i,j in rep_cycle_simplices], 
  ys = [X[[i,j],1] for i,j in rep_cycle_simplices], 
  line_color = 'blue', line_width = 3
)
p.line(x=X[ess_creator, 0], y=X[ess_creator, 1], line_color='red', line_width=3)
show(p)


# pred_simplices = [sx.Simplex(s[1]) for i, s in enumerate(K) if i < ess_index]

# %% 
b_true = np.ravel(R[:,ess_index].todense())
v_true = np.ravel(V[:,ess_index].todense())

# %% Get the boundary matrix 
from scipy.spatial.distance import pdist
diam_filter = sx.flag_filter(pdist(X))
ess_idx = list(sx.faces(S,1)).index(essential_creator)
pred_idx = diam_filter(sx.faces(S, 1)) <= diam_filter(essential_creator)
pred_idx[essential_idx] = False
pred_idx = np.flatnonzero(pred_idx)

## Get the 1-d boundary matrix
D = sx.boundary_matrix(S, p=1)

## Set the successors to 0
valid_cols = pred_idx[np.searchsorted(pred_idx, D.col)] == D.col
D.data[~valid_cols] = 0

## Get the chain for the creator of interest 
b = np.zeros(D.shape[1])
b[17] = 1
b[18] = -1

## Solve the linear system approximately or exactly
from scipy.sparse.linalg import spsolve, lsqr
lsqr(D, b)


from scipy.sparse import bmat, coo_array, csr_array
V_block = coo_array((sx.card(S,0),sx.card(S,0)))
E_block = coo_array((sx.card(S,0),sx.card(S,1)))
B = bmat([[V_block, D], [E_block.T, None]])
v = np.append(np.zeros(sx.card(S,0)), b)

spsolve(B, v)

res = lsqr(B, v, atol=0.0)
B @ np.zeros(B.shape[0])
lsqr(B, np.ones(len(v)))



# solve(A.todense(), b) # nope: matrix is singular (obviously)
# spsolve(A.tocsc(), b) # nan 