# %% Imports 
from itertools import *
from typing import * 
import networkx as nx
import numpy as np
import open3d as o3
from more_itertools import chunked, pairwise
from pbsig.betti import Sieve
from pbsig.linalg import eigvalsh_solver
from scipy.sparse.csgraph import floyd_warshall
from splex import *

# %% Bokeh configuration 
import bokeh
from bokeh.plotting import show, figure 
from bokeh.models import Range1d
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()

# %%  Data set 
from pbsig.datasets import noisy_circle
X = noisy_circle(150, n_noise=20, perturb=0.10, r=0.10)
p = figure(width=250, height=250)
p.scatter(*X.T)
show(p)

# %% Use the insertion radius of the greedy permuttion 
from pbsig.shape import landmarks
from pbsig.persistence import ph
pi, insert_rad = landmarks(X.T, k=len(X), diameter=True)

# %% Construct the 2-d rips compelx at the enclosing radius to true diagram
from pbsig.vis import figure_dgm
from splex.geometry import rips_complex, enclosing_radius
er = enclosing_radius(X)
S = rips_complex(X, radius=er, p=2)
K = rips_filtration(X, radius=er, p=2)
dgm = ph(K, engine="dionysus")

# %% Show the diagram 
p = figure_dgm(dgm[1])
show(p)


# %% Test sparse complexes
dgms_sparse = [ph(rips_filtration(X[pi[:i]], radius=er, p=2), engine="dionysus") for i in range(10, 60, 5)]
dgm_figs = []
for dgm in dgms_sparse:
  p = figure_dgm(dgm[1], width=200, height=200)
  p.x_range = Range1d(0, 1)
  p.y_range = Range1d(0, 1)
  dgm_figs.append(p)

show(row(*dgm_figs))


# %% Compute the weights for each vertex
# https://arxiv.org/pdf/1306.0039.pdf
# https://www.cs.purdue.edu/homes/tamaldey/book/CTDAbook/CTDAbook.pdf
# https://research.cs.queensu.ca/cccg2015/CCCG15-papers/01.pdf 
from scipy.spatial.distance import pdist
# eps = 0.1 # 0 < eps < 1
# def sparse_weight(X: np.ndarray, radii: np.ndarray, eps: float) -> Callable:
#   pd = pdist(X)
#   def _vertex_weight(i: int, alpha: float):
#     if (radii[i]/eps) >= alpha: return 0.0
#     if (radii[i]/(eps-eps**2)) <= alpha: return eps*alpha
#     return alpha - radii[i]/eps
  
#   def _weight(s: SimplexConvertible, alpha: float):
#     if dim(s) == 0: 
#       return _vertex_weight(s[0], alpha) 
#     elif dim(s) == 1:
#       return pd[rank_lex(s, n=len(X))] + _vertex_weight(s[0], alpha) + _vertex_weight(s[1], alpha)
#     else:
#       return np.max([_weight(f, alpha) for f in faces(s, p=1)])
#   return _weight

def sparse_weight(G: ComplexLike, L: np.ndarray, epsilon: float, alpha: float):
  """Sparse rips complex """
  assert epsilon > 0, "epsilon parameter must be positive"
  eps = ((1.0+epsilon)**2)/epsilon
  edges = np.array(faces(G, 1))
  edge_sparse_ind = np.flatnonzero(~((L[edges]*eps).max(axis=1) <= alpha))
  edge_weights = (L[edges[edge_sparse_ind]]).min(axis=1)*eps
  return edges[edge_sparse_ind], edge_weights

# %% Test sparsified complexes 
R = (0, 0.3, 0.6, 1.0)
ES_edges, ES_weight = sparse_weight(S, insert_rad, 0.90, er)

card(S,1)
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.90, er)[0])
S_sparse = simplicial_complex()
S_sparse.update([[i] for i in range(card(S,0))])
S_sparse.update(sparse_weight(S, insert_rad, 0.90, er)[0])



from gudhi import RipsComplex
S = RipsComplex(points=X, max_edge_length=er, sparse=0.90)
st = S.create_simplex_tree()

S_sparse = simplicial_complex([Simplex(s[0]) for s in st.get_simplices()])

p2 = figure_complex(S_sparse, pos=X)

# f_sparse = sparse_weight(X, insert_rad, 0.1)

# face_weights = np.array([f_sparse(s,) for s in faces(S)])


# %% Run exponential search 


# %%  Fix alpha, optimize epsilon
# TODO: insert points one at a time by non-zero additions to matvec operator...
from functools import partial
ir = np.array(insert_rad)[np.argsort(pi)] # 
wf = sparse_weight(X, ir, eps=1-1e-12)
wf = sparse_weight(X, ir, eps=0.00001)
K = filtration(S, partial(wf, alpha=er))

sum(np.array(list(K.indices())) <= 2*er)


from pbsig.vis import figure_dgm
dgm = ph(K, engine="dionysus")
p = figure_dgm(dgm[1])
show(p)


# %% 












# %% 
from metricspaces import MetricSpace
from greedypermutation import clarksongreedy, Point
point_set = [Point(x) for x in X]
M = MetricSpace(point_set)
G = list(clarksongreedy.greedy(M))
G_hashes = [hash(g) for g in G]
P_hashes = [hash(p) for p in point_set]


