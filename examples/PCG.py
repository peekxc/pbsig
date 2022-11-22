from typing import * 
import numpy as np 
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator
import networkx as nx 

from pbsig.persistence import boundary_matrix
from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1, uniform_S1
from pbsig.datasets import mpeg7
from pbsig.pht import pht_preprocess_pc
import primme

opts = dict(tol=1e-6, printLevel=0, return_eigenvectors=False, return_stats=True, return_history=True)
G = nx.connected_watts_strogatz_graph(130, k=4, p=0.20)
G = nx.fast_gnp_random_graph(250, p=0.05)

K = { 'vertices': np.array(list(G.nodes)), 'edges': np.array(list(G.edges)) }
X = pht_preprocess_pc(np.array(list(nx.fruchterman_reingold_layout(G).values())))

import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(5,5))
nx.draw(G, pos=X, ax=plt.axes(), node_size=15)

## Function to generate matrices
def weighted_graph_Laplacian(K: dict, v: ArrayLike):
  D1 = boundary_matrix(K, p = 1).tocsc().sorted_indices()
  fv = X @ np.array(v)
  fe = fv[K['edges']].max(axis=1)
  D1.data = np.sign(D1.data)*np.repeat(fe, 2)
  L = D1 @ D1.T
  return(L, D1)

L, D1 = weighted_graph_Laplacian(K, [1,0])

## Condition number 
# https://blogs.mathworks.com/cleve/2017/07/17/what-is-the-condition-number-of-a-matrix/
# ew = np.linalg.eigvalsh(L.todense() + 1.0*np.eye(L.shape[0]))
# max(ew)/min(ew)
# np.linalg.cond(L.todense() + 1.0*np.eye(L.shape[0]))
def graph_laplacian(G):
  from scipy.sparse import diags
  D = diags([G.degree(i) for i in range(G.number_of_nodes())])
  A = nx.adjacency_matrix(G)
  return D - A

def cond(A):
  S = np.linalg.svd(A, compute_uv = False)
  tol = S.max() * max(A.shape) * np.finfo(float).eps
  r = sum(S >= tol)
  return S.max()/S[r-1]

class Counter():
  def __init__(self): self.cc = 0
  def __call__(self, *args, **kwargs): self.cc += 1
  def __repr__(self): return f"Number of calls: {self.cc}"

## Note this should'nt be possible to solve as L is not invertible! 
from scipy.sparse.linalg import cg
x = np.random.uniform(size=L.shape[0], low=-0.50, high=0.50)
b = L @ x 
n_calls = Counter()
xs, info = cg(A=L.todense(), b=b, tol=1e-15, maxiter=1500, callback=n_calls)
max(abs(xs - x))

## Make the matrix positive definite and thus invertible!
from scipy.sparse import eye
LI = L + eye(L.shape[0])
b = LI @ x 
n_calls = Counter()
xs, info = cg(A=LI, b=b, tol=1e-15, maxiter=1500, callback=n_calls)
assert max(abs(xs - x)) <= 1e-14

## Let's see how condition number affects convergence
for eps in [10.0, 1.0, 0.10, 0.01, 0.001]:
  LI = L + eps*eye(L.shape[0])
  b = LI @ x 
  n_calls = Counter()
  xs, info = cg(A=LI, b=b, tol=1e-15, maxiter=1500, callback=n_calls)
  print(f"dim = {LI.shape[0]}, eps = {eps}, {n_calls}, Max residual: {max(abs(xs - x))}, condition number: {np.linalg.cond(LI.todense())}")

def benchmark_precond(L, Minv, n_iter = 10, tol=1e-15):
  for eps in [10.0, 1.0, 0.10, 0.01, 0.001]:
    LI = L + eps*eye(L.shape[0])
    mr, nc = 0, 0# avergae max residual 
    for i in range(n_iter):
      x = np.random.uniform(size=L.shape[0], low=-0.50, high=0.50)
      b = LI @ x 
      n_calls = Counter()
      xs, info = cg(A=LI, b=b, tol=tol, maxiter=1500, callback=n_calls, M=Minv)
      nc += n_calls.cc
      mr += max(abs(xs - x))
    print(f"dim = {LI.shape[0]}, eps = {eps}, (avg) # iterations {nc/n_iter}, (avg) max residual: {mr/n_iter}, L condition number: {np.linalg.cond(LI.todense())}")


## Different preconditioners
from pbsig.lsst import low_stretch_st
L, D1 = weighted_graph_Laplacian(K, [1,0])
G = nx.from_numpy_array((L - diags(L.diagonal())).todense())
# any([i == j for (i,j) in G.edges()])

from scipy.linalg import pinvh
T = low_stretch_st(G, "akpw", weighted=True) # use weighted !
GT = nx.from_numpy_matrix(T)
assert nx.is_connected(GT) and nx.is_tree(GT)
LT = graph_laplacian(nx.from_numpy_matrix(T))

w = cholesky(LT)

# T = low_stretch_st(G, "randishPrim", weighted=False)
# LT = graph_laplacian(nx.from_numpy_matrix(T))

cond(L.todense())
cond(pinvh(LT.todense()) @ L.todense())

# Idea from: http://www.lix.polytechnique.fr/~maks/papers/SEA_2015_draft.pdf
DG = diags(L.diagonal())
HT = DG - T

cond(L.todense())
cond(pinvh(L.todense()))
cond(pinvh(L.todense()) @ L.todense())
cond(diags(1/L.diagonal()) @ L.todense()) # this seems to be the best in terms of condition number
cond(pinvh(HT) @ L.todense())

max(np.linalg.svd(pinvh(LT.todense())@L, compute_uv=False)) >= 1
## Upper bound on Lh^+ Lg is nm = G.number_of_edges() * G.number_of_nodes()
sum(np.diag(pinvh(LT.todense()) @ L.todense()))
sum(np.diag(pinvh(HT) @ L.todense()))


benchmark_precond(L, Minv = np.eye(L.shape[0]))
benchmark_precond(L, Minv = pinvh(L.todense()))
benchmark_precond(L, Minv = pinvh(LT.todense()))
benchmark_precond(L, Minv = diags(1/L.diagonal()))
benchmark_precond(L, Minv = pinvh(HT))
benchmark_precond(L, Minv = CholSolver(L, 0.001))
benchmark_precond(L, Minv = CholSolver(LT, 0.0))

# b: array([390, 422, 579, 598, 802, 659])
# array([161, 432, 387, 330,  96,  81])
from scipy.sparse import eye
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from sksparse.cholmod import cholesky
factor = cholesky(L + 0.01*eye(L.shape[0]))
x = factor(b)

class CholSolver(LinearOperator):
  def __init__(self, L, eps: float):
    self.dtype = np.dtype(float)
    self.shape = L.shape
    self.eps = eps
    self.factor = cholesky(L + self.eps*eye(L.shape[0]))
  def _matvec(self, x):
    return self.factor(x)





LI = L + 1.0*eye(L.shape[0])
b = LI @ x 

n_calls = Counter()
xs, info = cg(A=LI, b=b, tol=1e-15, maxiter=1500, callback=n_calls)
print(f"dim = {LI.shape[0]}, eps = {eps}, {n_calls}, Max residual: {max(abs(xs - x))}, condition number: {np.linalg.cond(LI.todense())}")

n_calls = Counter()
xs, info = cg(A=LI, b=b, tol=1e-15, maxiter=1500, callback=n_calls, M=T)
print(f"dim = {LI.shape[0]}, eps = {eps}, {n_calls}, Max residual: {max(abs(xs - x))}, condition number: {np.linalg.cond(LI.todense())}")

# 1/np.diag(np.diag(L.todense())) + T
np.linalg.cond(T @ LI)
np.linalg.cond(LI.todense())
np.linalg.cond((np.eye(T.shape[0]) + T))
np.linalg.cond((np.eye(T.shape[0]) + T) @ LI)


#from scipy.sparse import 
r = np.linalg.matrix_rank(L.todense())
s = np.linalg.svd(L.A, compute_uv = False)
s[0]/s[r-1]
LT = graph_laplacian(TG)
st = np.linalg.svd(LT.A, compute_uv = False)
st[0]/st[r-1]








## Randomized kruskal's 
import networkx as nx
from networkx.utils import UnionFind
nv = G.number_of_nodes()
ds = UnionFind(range(nv))
E = np.array(list(G.edges()))

# Algorithm for getting spectral sparsifier: https://people.orie.cornell.edu/dpw/orie6334/lecture18.pdf
# 0 < eps < 1 
# Effective resistance calculation: https://www.cs.yale.edu/homes/spielman/561/lect14-18.pdf

e_ind = np.array(list(range(E.shape[0])))
w = np.repeat(1, E.shape[0])/E.shape[0]
normalize = lambda x: x/sum(x)
T = []
while len(list(ds.to_sets())) > 1:
  ii = np.random.choice(e_ind, p=normalize(w[e_ind]))
  i,j = E[ii,:]
  if ds.parents[i] == ds.parents[j]:
    continue
  else: 
    ds.union(i,j)
    T.append((i,j))
  e_ind = np.setdiff1d(e_ind, ii)


# nfft?


def edge_iter(A):
  for i,j in zip(*A.nonzero()):
    if i < j: 
      yield (i,j)

## Given a graph Laplacian, calculate the effective resistance of every edge
def effective_resistance(L: ArrayLike, method: str = "pinv"):
  eff_resist = lambda x: x
  if method == "psvd_sqrt":
    u,s,vt = np.linalg.svd(L.todense(), hermitian=True)
    tol = s.max() * max(L.shape) * np.finfo(float).eps
    r = sum(s > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/np.sqrt(s[:(r-1)])) @ u[:,:(r-1)].T
    eff_resist = lambda x: np.linalg.norm(LS @ x)**2
  elif method == "pinv":
    from scipy.linalg import pinvh
    LP, r = pinvh(L.todense(), return_rank=True)
    eff_resist = lambda x: (x.T @ LP @ x).item()
  elif method == "psvd":
    u,s,vt = np.linalg.svd(L.todense(), hermitian=True)
    tol = s.max() * max(L.shape) * np.finfo(float).eps
    r = np.sum(abs(s) > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/s[:(r-1)]) @ u[:,:(r-1)].T
    eff_resist = lambda x: (x.T @ LS @ x).item()
  elif method == "randomized":
    raise NotImplementedError("Haven't done yet")
  else: 
    raise ValueError("Unknown method")
  nv = L.shape[0]
  cv, er = np.zeros(shape=(nv,1)), []
  for (i,j) in edge_iter(L):
    cv[i] = 1
    cv[j] = -1
    er.append(eff_resist(cv))
    cv[i] = cv[j] = 0
  return np.array(er)

## Spectral sparsifer 
def spectral_sparsifier(G, eps = 0.80):
  assert eps >= 0 and eps <= 1
  import networkx as nx 
  from scipy.sparse import diags
  D = diags(list(dict(nx.degree(G)).values()))
  A = nx.adjacency_matrix(G)
  L = D - A
  nv, ne = G.number_of_nodes(), G.number_of_edges()
  eps = (1-eps)*(1/np.sqrt(nv)) + eps*1.0
  E = np.array(list(G.edges()), dtype=int)
  w = np.zeros(shape=ne)
  nl = int(6*(nv-1)*np.log(nv)/(eps**2))
  er = effective_resistance(L)
  er /= sum(er)
  es = np.random.choice(range(ne), size=nl, replace=True, p=er) # edge samples 
  #for s in es: w[s] += (nv-1)/(nl*er[s])
  return E[np.unique(es),:]








(L[:,[0]].T @ L @ L[:,[0]]).data[0]

from scipy.sparse.linalg import eigsh
ew, ev = eigsh(L)
LS = np.zeros(shape=L.shape)
for i, l, in enumerate(ew):
  if abs(l) > 1e-12:
    LS += (1/l)*(ev[:,[i]] @ ev[:,[i]].T)
LS = np.linalg.pinv(L.todense())  



import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(5,5))
# nx.draw(G, pos=X, ax=plt.axes(), node_color=er, node_size=15)
# nx.draw(G, pos=X, ax=plt.axes(), node_color=w, node_size=15)



re = np.vectorize(lambda x, eps: x**2 /(x**2 + eps))
re = np.vectorize(lambda x, eps: x /(x**2 + eps**2)**(1/2))
re = np.vectorize(lambda x, eps: 0 if eps == 0 else 1-np.exp(-(x/eps)))

x = np.linspace(-1, 1, 1000)
EPS = np.quantile(np.linspace(np.log(1), np.log(2), 1000), [0, 0.25, 0.50, 0.75, 1.0])
EPS = [0, 0.05, 0.10, 0.25, 0.50, 0.75]
for eps in EPS:
  plt.plot(x, re(abs(x), eps))