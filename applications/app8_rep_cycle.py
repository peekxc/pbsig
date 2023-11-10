# %%
import numpy as np
from typing import * 
from scipy.sparse import spmatrix
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
X = noisy_circle(20, n_noise=0, perturb=0.05) ## 80, 30, 0.15 
S = rips_complex(X, radius=0.75, p=1)

# %% Peek at the compelx 
show(figure_complex(S, X))

#%% Get the rips diagram 
from pbsig.persistence import ph
K = sx.rips_filtration(X, radius=0.75, p=2) ## need triangles to collapse 
dgms = ph(K, simplex_pairs=True)
assert len(dgms[1]) == 1
essential_creator = np.take(dgms[1]['creators'], 0)

R, V = ph(K, output="RV")
np.flatnonzero([sx.Simplex(s) == sx.Simplex((17,18)) for s in K])

ess_idx_filtered = np.take(np.flatnonzero(np.all(np.array(list(sx.faces(K,1))) == np.array([17,18]), axis=1)), 0)

b_true = np.ravel(R[:,ess_idx_filtered].todense())
v_true = np.ravel(V[:,ess_idx_filtered].todense())

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