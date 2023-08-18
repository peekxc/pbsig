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

## Needs: 
##  1. Sequence with __len__ + __getitem__() support
##  2. target type with __eq__ and __le__ support
def binary_search(seq: Sequence, target: Any):
  left, right = 0, len(seq) - 1
  while left <= right:
    mid = left + (right - left) // 2
    val = seq[mid]
    if val == target:
      return mid
    left, right = (mid + 1, right) if val < target else (left, mid - 1)
  return None # Element not found
 
assert np.all([binary_search(x, x[i]) == i for i in range(len(x))])

## TODO: augment to work only for '<=' or better for arbitrary predicate
## Also augment to work with binary search as well  
# goal is to find the index 'i' right before the predicate returns False, i.e.
# TRUE, TRUE, TRUE, ..., TRUE, TRUE,  FALSE, FALSE, ..., FALSE
# ------------------------------(i)^--(i+1)^------------------ 
def exponential_search(seq: Sequence, target: Any, l: int = 0):
  print(f"({l})")
  if seq[l] == v:
    return l
  n = 1
  while (l+n) < len(seq) and seq[l+n] <= v:
    #print(l+n)
    n *= 2
  l = min(l + (n // 2), len(seq) - 1) # min(l+n, len(arr) - 1)
  return exponential_search(seq, v, l)

assert np.all([exponential_search(x, x[i]) == i for i in range(len(x))])

exponential_search(np.array([True, True, True, True, False, False]), False)

## Also augment to work with binary search as well  
# goal is to find the index 'i' right before the predicate returns False, i.e.
# TRUE, TRUE, TRUE, ..., TRUE, TRUE,  FALSE, FALSE, ..., FALSE
# ------------------------------(i)^--(i+1)^------------------ 
def first_true(seq: Sequence):
  left, right = 0, len(seq) - 1
  val = False
  while left < right:
    mid = left + (right - left) // 2
    val = seq[mid]
    print(mid)
    if val is False: 
      left = mid + 1
    else:
      if mid == 0 or seq[mid - 1] is False:
        return mid
      right = mid - 1
  return left if val else -1


def first_true_exp(seq: Sequence, lb: int = 0, ub: int = None, k: int = 1):
  lb = min(lb, len(seq) - 1)
  ub = len(seq) - 1 if ub is None else max(ub, len(seq) - 1)
  # print(lb)
  val = seq[lb]
  print(f"E: val: {val}, ({lb}, {ub})")
  # print(f"lb: {lb}")
  i = 0
  while not(val) and lb < ub: 
    lb = lb + 2**i
    if lb > ub: 
      lb, ub = (lb - 2**i) + 1, ub 
      return first_true_exp(seq, lb, ub)   
    else: 
      i += 1
      val = seq[lb]
      print(f"E: lb: {lb}")
  lb, ub = max(min(lb, ub), 0), min(max(lb, ub), len(seq) - 1)
  print(f"val: {val} /({not(val)}), ({lb}, {ub}) i={i}, ({lb == (ub-1)})")
  if val and i <= 1: # (lb == ub)
    return lb
  elif not val and lb == ub:
    return -1
  else:
    # print(f"WHAT: {val}, {(i <= 1 or lb == ub)}")
    # lb, ub = max((lb - 2**(i-1)), lb + 1), lb + 1
    lb, ub = (lb - 2**(i-1)) + 1, ub
    return first_true_exp(seq, lb, ub)    

L = [False]*1345 + [True]*1500

z = np.zeros(1500, dtype=bool)
assert first_true_exp(z) == -1
for i in range(len(z)-1):
  z[-(i+1)] = True 
  assert first_true_exp(z) == (len(z) - i - 1)

print(first_true_exp(z, lb=0, ub=len(L)-1))


## This works
def first_true_bin(seq: Sequence, lb: int, ub: int):
  while lb <= ub:
    mid = lb + (ub - lb) // 2
    val = seq[mid]
    if val and (mid == 0 or not seq[mid - 1]):
      return mid
    lb,ub = (lb, mid - 1) if val else (mid + 1, ub)
  return -1

z = np.zeros(1500, dtype=bool)
first_true_bin(z, 0, len(z)-1)
for i in range(len(z)-1):
  z[-(i+1)] = True 
  assert first_true_bin(z, 0, len(z)-1) == (len(z) - i - 1)

first_true_bin(L, 0, len(L)-1)

def binary_search_first(seq: Sequence, lb: int = 0, ub: int = None):
  lb, ub = max(0, lb), len(seq) - 1 if ub is None else min(ub, len(seq) - 1)
  while lb <= ub:
    mid = lb + (ub - lb) // 2
    val = seq[mid]
    if val:
      return mid
    left, right = (mid + 1, right) if val < target else (left, mid - 1)
  return None # Element not found

# if (val and lb == 0) or (val and lb == ub): 
#   return lb 
# elif val == True and lb != ub: 
#   return lb 
# else: 
#   lb = lb * 2
#   first_true_exp


# def first_true_exp(seq: Sequence, ub: int = 1):
#   print(ub)
#   if ub >= len(seq): ## we know its in the last half then
#     lb = ub // 2

#   if seq[ub] is True:
#     if ub == 0 or seq[ub - 1] is False:
#       return ub
#     else:
#       return first_true_exp(seq, ub * 2)
#   else:
#     return first_true_exp(seq, ub * 2 + 1)
    
L = [False]*1345 + [True]*15000

first_true(L)


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
# from pbsig.shape import landmarks
# mesh_loader = pose_meshes(simplify=20000, which=["elephant"])
# elephant_mesh = mesh_loader[0]
# elephant_mesh.euler_poincare_characteristic() # 2 
# elephant_mesh.is_watertight() # False 

# elephant_pos = np.asarray(elephant_mesh.vertices)
# lm_ind, lm_radii = landmarks(elephant_pos, k=len(elephant_pos))

def geodesics(mesh):
  from scipy.sparse.csgraph import floyd_warshall
  from scipy.spatial.distance import squareform, cdist
  import networkx as nx
  vertices = np.asarray(mesh.vertices)
  mesh.compute_adjacency_list()
  G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(mesh.adjacency_list) })
  A = nx.adjacency_matrix(G).tocoo()
  A.data = np.linalg.norm(vertices[A.row] - vertices[A.col], axis=1)
  #A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
  AG = floyd_warshall(A.tocsr())
  return AG

## Compute ground-truth eccentricities for edges and vertices
## Does the mesh eccentricity even show convergent behavior
mesh_loader = pose_meshes(simplify=8000, which=["elephant"]) # 8k triangles takes ~ 40 seconds
elephant_mesh = mesh_loader[0]
A = geodesics(elephant_mesh)
mesh_ecc = A.max(axis=1)
ecc_weight = lower_star_weight(mesh_ecc)
K_ecc = filtration(simplicial_complex(np.asarray(elephant_mesh.triangles)), ecc_weight)

## Final boxes: 
## card([1.48, 1.68] x [1.70, 1.88]) == 4   ()
## card([1.12, 1.28] x [1.36, 1.56]) == 2
## card([1.0, 1.2] x [2.0, 2.2]) == 1       (torso)
## 
from pbsig.persistence import ph
from pbsig.vis import figure_dgm
dgm = ph(K_ecc, engine="dionysus")
p = figure_dgm(dgm[1], tools="box_select")
show(p)

# %% Run exponential search 
from pbsig.betti import Sieve
from pbsig.linalg import spectral_rank

mesh_loader = pose_meshes(simplify=100, which=["elephant"]) # 8k triangles takes ~ 40 seconds
elephant_mesh = mesh_loader[0]
A = geodesics(elephant_mesh)
mesh_ecc = A.max(axis=1)
ecc_weight = lower_star_weight(mesh_ecc)
# K_ecc = filtration(simplicial_complex(np.asarray(elephant_mesh.triangles)), ecc_weight)


S = simplicial_complex(np.asarray(elephant_mesh.triangles))
sieve = Sieve(S, family=[ecc_weight], p = 1)
sieve.pattern = np.array([[1.48, 1.68, 1.70, 1.88], [1.12, 1.28, 1.36, 1.56], [1.0, 1.2, 2.0, 2.2]])
# sieve.pattern = np.array([[1.0, 1.2, 2.0, 2.2]])
# sieve.family
# sieve.project(1.0, 1.2, 0.0, sieve.family[0])

sieve.sift()
np.ravel(sieve.summarize(f=spectral_rank))

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


