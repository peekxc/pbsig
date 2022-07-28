import numpy as np
from . import boundary
from .persistence import *
from .betti import *

def plot_mesh2D(X: ArrayLike, edges: ArrayLike, triangles: ArrayLike, **kwargs):
  import matplotlib.pyplot as plt 
  fig_kwargs = kwargs.get('figure', dict(figsize=(4,4), dpi=150))
  fig = plt.figure(**fig_kwargs)
  ax = fig.gca()
  ax.scatter(*X.T, s=4.30, c='black')
  ax.set_aspect('equal')
  for e in edges: ax.plot(X[e,0], X[e,1], c='black', linewidth=0.80)
  for t in triangles: ax.fill(X[t,0], X[t,1], c='#c8c8c8b3')
  return(fig, ax)

def rotate_S1(X: ArrayLike, n: int = 10):
  """ 
  Create an generator that returns the inner product of points in 'X' along 'n' different directions on the unit circle. 
  This is a key subroutine to the PHT. 

  Params: 
    X := (m x 2) ndarray of points 
    n := number of directions to discretize S1 by (uniformly)
  
  Returns: 
    Generator which yields a tuple (fv, v) where: 
    fv := m-length ndarray of inner products of X with unit vector 'v' 
    v := the unit vector 'X' was projected onto
  """
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)
  for x,y in zip(np.sin(theta), np.cos(theta)):
    v = np.array([x,y])[:,np.newaxis]
    yield ((X @ v).flatten(), v)

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