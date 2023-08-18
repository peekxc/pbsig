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
X = noisy_circle(50, n_noise=10, perturb=0.10, r=0.10)
p = figure(width=250, height=250)
p.scatter(*X.T)
show(p)

# %% Use the insertion radius of the greedy permuttion 
from pbsig.shape import landmarks
from pbsig.persistence import ph
pi, insert_rad = landmarks(X, k=len(X), diameter=True)

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

# %% Test what greedypermutation's prefix points look like
dgms_sparse = [ph(rips_filtration(X[pi[:i]], radius=er, p=2), engine="dionysus") for i in range(10, 60, 5)]
dgm_figs = []
for dgm in dgms_sparse:
  p = figure_dgm(dgm[1], width=200, height=200)
  p.x_range = Range1d(0, 1.1)
  p.y_range = Range1d(0, 1.1)
  dgm_figs.append(p)

show(row(*dgm_figs))

# %% Simple exponential search 
x = np.sort(np.random.choice(range(1500), size=340, replace=False))
v = 78

## TODO: augment to work only for '<=' or better for arbitrary predicate
## Also augment to work with binary search as well  
# goal is to find the index 'i' right before the predicate returns False, i.e.
# TRUE, TRUE, TRUE, ..., TRUE, TRUE,  FALSE, FALSE, ..., FALSE
# ------------------------------(i)^--(i+1)^------------------ 
def exponential_search(arr: np.ndarray, v: Any, l: int = 0):
  #print(f"({l})")
  if arr[l] == v:
    return l
  n = 1
  while (l+n) < len(arr) and arr[l+n] <= v:
    #print(l+n)
    n *= 2
  l = (l + (n // 2)) # min(l+n, len(arr) - 1)
  return exponential_search(arr, v, l)

assert np.all([exponential_search(x, x[i]) == i for i in range(len(x))])


# %% Problem: using landmark points for subsampled rips is still a rips complex, which is intractable....
# Need an alternative complex or data set... maybe delaunay...
# Or just use the given sparse mesh. Take an elephant, choose landmarks/greedypermutation, then...
# 1) Consider problem of *generating* a sparse filtered complex from a *point set alone*---start with the true diagram, and 
# then use exp search on landmark points until a given multiplicity is satisfied (say, w/ geodesic eccentricity filter + delaunay complex)
# UPDATE: this seems not feasible as delaunay ecc wont generate the right dgm as the structured mesh
# 2) Reverse problem: You have a mesh + it's true diagram (from mesh!), you wish to sparsify (say w/ quadric simplification) 
# as much as possible, but under the constraint it preserves a given topology. Again, eccentricity + elephant constraint. 
# 
# could also *perhaps* formulate a rips/based or spectra based objective for sparsification

# %% (1) Load mesh and do floyd warshall to get geodesic distances
from pbsig.datasets import pose_meshes
from pbsig.shape import landmarks
mesh_loader = pose_meshes(simplify=4000)
elephant_mesh = mesh_loader[6]
# elephant_mesh.euler_poincare_characteristic() # 2 
# elephant_mesh.is_watertight() # False 

elephant_pos = np.asarray(elephant_mesh.vertices)
lm_ind, lm_radii = landmarks(elephant_pos, k=len(elephant_pos))

from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import squareform, cdist
import networkx as nx
elephant_mesh.compute_adjacency_list()
G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(elephant_mesh.adjacency_list) })
A = nx.adjacency_matrix(G).tocoo()
A.data = np.linalg.norm(elephant_pos[A.row] - elephant_pos[A.col], axis=1)
#A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
AG = floyd_warshall(A.tocsr())

## Compute eccentricities for edges and vertices
n = elephant_pos.shape[0]
mesh_ecc = AG.max(axis=1)
ecc_weight = lower_star_weight(mesh_ecc)
K_ecc = filtration(simplicial_complex(np.asarray(elephant_mesh.triangles)), ecc_weight)


# %% (2) Does delaunay recover this?
from pbsig.persistence import ph
from pbsig.vis import figure_dgm
dgm = ph(K_ecc, engine="dionysus")

p = figure_dgm(dgm[1])
show(p)

from pbsig.linalg import up_laplacian
from scipy.sparse import diags
S = delaunay_complex(np.asarray(elephant_mesh.vertices))

L = up_laplacian(S, p=0)
A = L - diags(L.diagonal())
A = A.tocoo()
A.data = np.linalg.norm(elephant_pos[A.row] - elephant_pos[A.col], axis=1)
#A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
AG = floyd_warshall(A.tocsr())
mesh_ecc = AG.max(axis=1)
ecc_weight = lower_star_weight(mesh_ecc)


K_ecc = filtration(S, ecc_weight)
dgm = ph(K_ecc, engine="dionysus")
p = figure_dgm(dgm[1])
show(p)





# %% Precompute eccentricities
mesh_ecc = [squareform(gd).max(axis=1) for gd in mesh_geodesics]
ecc_f = [[max(mesh_ecc[cc][i], mesh_ecc[cc][j]) for (i,j) in combinations(range(X.shape[0]), 2)] for cc, (X, mesh) in enumerate(meshes)]
eff_f = [np.array(ecc) for ecc in ecc_f]


# %% Use gudhi's sparse rips 



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

def sparse_rips_filtration(X: ArrayLike, L: np.ndarray, epsilon: float, p: int = 1, max_radius: float = np.inf, remove_singletons: bool = True):
  """Sparse rips complex """
  assert epsilon > 0, "epsilon parameter must be positive"

  d = pdist(X)
  # np.digitize(x, bins=[0.33, 0.66, 1])

  ## Step 0: Make a vectorized edge birth time function 
  def edge_birth_time(edges: ArrayLike, eps: float, n: int) -> np.ndarray:
    ## Algorithm 3 from: https://arxiv.org/pdf/1506.03797.pdf
    edge_ind = np.array(rank_combs(edges, order='lex', n=n))
    L_min = L[edges].min(axis=1)
    birth_time = np.ones(len(edges))*np.inf # infinity <=> don't include in complex
    cond0 = d[edge_ind] <= 2 * L_min * (1+eps)/eps
    cond1 = d[edge_ind] <= L[edges].sum(axis=1) * (1+eps)/eps
    birth_time[cond1] = d[edge_ind[cond1]] - L_min[cond1]*(1+eps)/eps
    birth_time[cond0] = d[edge_ind[cond0]] / 2
    return birth_time

  ## Step 1: Construct the sparse graph of the input point cloud or metric X 
  n = len(X)
  all_edges = np.array(list(combinations(range(n), 2))) ## todo: add option to use kd tree or something
  edge_fv = edge_birth_time(all_edges, epsilon, n)
  S_sparse = simplicial_complex(chain([[i] for i in range(n)], all_edges[edge_fv != np.inf]))

  ## Step 2: Perform a sparse k-expansion, if 2-simplices or higher needed 
  if p > 1: 
    pass 


  # eps = ((1.0-epsilon)/2) * epsilon # gudhi's formulation
  # edges = np.array(faces(G, 1))
  # edge_weights = (L[edges]).max(axis=1)
  # edge_sparse_ind = np.flatnonzero(edge_weights >= alpha)
  # edge_weights = (L[edges[edge_sparse_ind]]).min(axis=1)*eps
  # return edges[edge_sparse_ind], edge_weights

# ([0, 1], 0.14069064627395436) == pdist(X[[0,1],:])

# %% Test sparsified complexes 
from gudhi import RipsComplex
S_rips = RipsComplex(points=X, max_edge_length=er/2, sparse=0.90)
st = S_rips.create_simplex_tree()
S_sparse = simplicial_complex([Simplex(s[0]) for s in st.get_simplices()])
#  If `block_simplex` returns true, the simplex is removed, otherwise it is kept. 

ES_edges, ES_weight = sparse_weight(S, insert_rad, 0.90, 2*er)

card(S,1)
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.90, 2*er)[0])
S_sparse = simplicial_complex()
S_sparse.update([[i] for i in range(card(S,0))])
S_sparse.update(sparse_weight(S, insert_rad, 0.90, er)[0])






# %% Show the sparsified complex
from pbsig.vis import figure_complex
p = figure_complex(S_sparse, pos=X)
show(p)

# %% Run exponential search 
R = (0, 0.3, 0.6, 1.0)


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


