from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
#from . import boundary

# from .ext import boundary
# from . import _boundary
# from . import _boundary
from .persistence import *
from .betti import *


autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike'
}

def plot_dgm(dgm: ArrayLike):
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib.patches import Polygon
  from matplotlib.collections import PatchCollection
  fig = plt.figure(figsize=(2,2), dpi=220)
  ax = fig.gca()
  ymax = max(filter(lambda x: x != np.inf, dgm.flatten()))
  dgm[dgm[:,1] == np.inf, 1] = ymax*1.05
  ax.scatter(*dgm.T, s=0.55, c='red')
  ax.set_xlim(0.0, 1.10*ymax)
  ax.set_ylim(0.0, 1.10*ymax)
  ax.set_aspect('equal')
  p = Polygon(np.array([[0,0], [1.10*ymax, 0], [1.10*ymax, 1.10*ymax]]), True, color="#808080")
  P = PatchCollection([p], alpha=0.4, color="#808080")
  formatter = "{:.2f}".format
  tick_loc = np.linspace(0.0, ymax, 6, endpoint=False)
  ax.add_collection(P)
  ax.set_yticks(np.append(tick_loc, 1.05*ymax), [formatter(x) for x in tick_loc]+['inf'])
  ax.tick_params(axis='both', which='major', labelsize=3)
  ax.plot([0.0, 1.05*ymax], [1.05*ymax, 1.05*ymax], color='gray', linestyle='dashed', linewidth=0.50)
  return(fig, ax)

def plot_mesh2D(X: ArrayLike, edges: ArrayLike, triangles: ArrayLike, labels: bool = False, **kwargs):
  import matplotlib.pyplot as plt 
  fig_kwargs = kwargs.get('figure', dict(figsize=(4,4), dpi=150))
  fig = plt.figure(**fig_kwargs)
  ax = fig.gca()
  ax.set_aspect('equal')
  for e in edges: ax.plot(X[e,0], X[e,1], c='black', linewidth=0.80)
  for t in triangles: ax.fill(X[t,0], X[t,1], c='#c8c8c8b3')
  ax.scatter(*X.T, s=4.30, c='black')
  if labels:
    for i, xy in enumerate(X): ax.annotate(i, xy)
  return(fig, ax)

def uniform_S1(n: int = 10):
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)+(np.pi/2)
  for x,y in zip(np.cos(theta), np.sin(theta)):
    yield np.array([x,y])


def rotate_S1(X: ArrayLike, n: int = 10, include_direction: bool = True):
  """ 
  Create an generator that returns the inner product of points in 'X' along 'n' different directions on the unit circle, 
  starting from the vector point up (v = (0, 1))

  Params: 
    X := (m x 2) ndarray of points 
    n := number of directions to discretize S1 by (uniformly)
  
  Returns: 
    Generator which yields a tuple (fv, v) where: 
    fv := m-length ndarray of inner products of X with unit vector 'v' 
    v := the unit vector 'X' was projected onto
  """
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)+(np.pi/2)
  for x,y in zip(np.cos(theta), np.sin(theta)):
    v = np.array([x,y])[:,np.newaxis]
    if include_direction:
      yield ((X @ v).flatten(), v)
    else: 
      yield (X @ v).flatten()

## From: https://stackoverflow.com/a/4696026/6912436
def rfft_xcorr(x, y):
  M = len(x) + len(y) - 1
  N = 2 ** int(np.ceil(np.log2(M)))
  X = np.fft.rfft(x, N)
  Y = np.fft.rfft(y, N)
  cxy = np.fft.irfft(X * np.conj(Y))
  cxy = np.hstack((cxy[:len(x)], cxy[N-len(y)+1:]))
  return cxy

## From: https://stackoverflow.com/a/4696026/6912436
def match(x, ref):
  cxy = rfft_xcorr(x, ref)
  index = np.argmax(cxy)
  return index if index < len(x) else index - len(cxy)

def phase_align(s1: Sequence, s2: Sequence):
  ind = match(s2, s1)
  offsets = np.fromiter(range(-5, 5), dtype=int)
  r_ind = np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in offsets])
  return(np.roll(s1, ind+offsets[r_ind]))

def one_parameter_family():
  return(0)

def random_2d():
  from itertools import combinations
  import networkx as nx
  n = 50
  G = nx.fast_gnp_random_graph(n, 0.25)
  X = np.vstack(list(nx.spring_layout(G).values()))
  edges = pbsig.rank_combs(G.edges(), n=50, k=2)

  k_cliques = lambda C: combinations(C, 3)
  triangles = np.concatenate([pbsig.rank_combs(k_cliques(c), n=50, k=3) for c in nx.find_cliques(G)])

  theta = np.linspace(0, 2*np.pi, 100, endpoint=True)


  cvec = lambda x: np.array(x)[:,np.newaxis]
  F = [X @ cvec(v) for v in zip(np.sin(theta), np.cos(theta))]
  
  import matplotlib.pyplot as plt
  for i, f in enumerate(F):
    plt.scatter(np.repeat(i, len(f)), f, s=0.05)
  np.hstack(f)

  from scipy.interpolate import CubicSpline
  

  FM = np.hstack(F)
  FP = [CubicSpline(theta, FM[i,:], bc_type='periodic') for i in range(FM.shape[0])]
  cutoff = 0.50 

  for i in range(20):
    plt.plot(theta, np.array([FP[i](a) for a in theta]))

  ## TODO: estimate maximum number of simplices for pre-allocation
  plt.plot(theta, np.array([FP[0](a) for a in theta]))
  in_circle = lambda x: x >= 0.0 and x < 2*np.pi
  crit_events = [np.array(list(filter(in_circle, fp.solve(cutoff)))) for fp in FP]
  crit_events = np.concatenate([np.vstack((np.repeat(i, len(ce)), ce)).T for i, ce in enumerate(crit_events)])
  plt.scatter(w, FP[0](w))

  ## Preprocess the simplex 'additions' + 'deletions' via the cutoff as changes to the 
  ## sparse boundary matrices (in CSC?)
  ## TODO: bound upper-envelope simplex filtration values with davenport-shinzel sequence
  f = F[0]
  is_active = lambda s: bool(max(f[s]) <= cutoff)
  active_vertices = list(range(n))
  active_edges = [(i,j) for (i,j) in combinations(active_vertices, 2) if is_active([i,j])]
  active_triangles = [(i,j,k) for (i,j,k) in combinations(active_vertices, 3) if is_active([i,j,k])]

  from scipy.spatial import Delaunay
  D = Delaunay(X)
  triangles = D.simplices
  active_triangles = np.array(list(filter(is_active, triangles)))
  from pbsig.betti import edges_from_triangles
  active_edges = edges_from_triangles(active_triangles, n)

  ## when does a *simplex* go from active to inactive to active?
  v_ids = crit_events[:,0].astype(int)
  tri = [1,2,3]
  ce = np.sort(crit_events[[i in tri for i in v_ids], 1])
  # valid_ce = abs(np.max(np.array([FP[i](ce) for i in tri]), axis=0) - cutoff) < 10*np.finfo(float).resolution
  
  ## 
  ind = np.argmax(np.array([FP[i](ce) for i in tri]), axis=0)
  ce_type = np.array([np.sign(FP[i].derivative(1)(x)) for i,x in zip(ind, ce)]).astype(int) # 1 == turning off
  


  for i in tri: 
    plt.plot(theta, FP[i](theta))
    plt.scatter(ce, FP[i](ce))

  ## Convert 
  ## Can we have a _fast_ sparse matvec from a filtration?
  ## possibly w/ laplacian 


  return(2)





# from itertools import *
# d, n = 8, 3

# def ds(d, n):
#   S = [[1]]
#   co, ci = 0, 0
#   for s in S:
#     for v in range(1,n+1):
#       g = (w[2:] != w[:-2] for w in combinations(s+[v],d+1))
#       if (v != s[-1]) * all(g):
#         S.append(s+[v])
#         #print("k")
#       ci += 1
#     co += 1
#   return(ci, co)


# print(S[-1]) 