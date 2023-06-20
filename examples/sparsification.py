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

# %% 
from pbsig.datasets import noisy_circle
X = noisy_circle(50, n_noise=20, perturb=0.10, r=0.10)
p = figure(width=250, height=250)
p.scatter(*X.T)
show(p)

# %% Use the insertion radius of the greedy permuttion 
from pbsig.shape import landmarks
from pbsig.persistence import ph
pi, insert_rad = landmarks(X.T, k=len(X), diameter=True)

# %% Construct the 2-d rips compelx at some scale
from splex.geometry import rips_complex, enclosing_radius
er = enclosing_radius(X)
S = rips_complex(X, radius=er, p=2)

# %% Compute the weights for each vertex
from scipy.spatial.distance import pdist
eps = 0.1 # 0 < eps < 1
def sparse_weight(X: np.ndarray, radii: np.ndarray, eps: float) -> Callable:
  pd = pdist(X)
  def _vertex_weight(index: int, alpha: float):
    if (radii[index]/eps) >= alpha: return 0.0
    if (radii[index]/(eps-eps**2)) <= alpha: return eps*alpha
    return alpha - radii[index]/eps
  
  def _weight(s: SimplexConvertible, alpha: float):
    if dim(s) == 0: 
      return 0 
    elif dim(s) == 1:
      return pd[rank_lex(s, n=len(X))] + _vertex_weight(s[0], alpha) + _vertex_weight(s[1], alpha)
    else:
      return np.max([_weight(f, alpha) for f in faces(s, p=1)])
  return _weight

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


