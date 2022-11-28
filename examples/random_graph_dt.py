from typing import * 
import numpy as np 
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator
import networkx as nx 
import primme

from pbsig.persistence import boundary_matrix
from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1, uniform_S1
from pbsig.datasets import mpeg7
from pbsig.pht import pht_preprocess_pc

default_opts = dict(tol=1e-6, printLevel=0, return_eigenvectors=False, return_stats=True, return_history=True)

# G = nx.connected_watts_strogatz_graph(130, k=7, p=0.20)
G = nx.newman_watts_strogatz_graph(n=130, k=10, p=0.20)
# G = nx.random_geometric_graph(n=130, radius=0.10, dim=3)
# G = nx.erdos_renyi_graph(n=100, p=0.10)

K = { 'vertices': np.array(list(G.nodes)), 'edges': np.array(list(G.edges)) }
X = pht_preprocess_pc(np.array(list(nx.fruchterman_reingold_layout(G).values())))

# fig = plt.figure(figsize=(5,5))
# nx.draw(G, pos=X, ax=plt.axes(), node_size=15)

## Function to generate matrices
def weighted_graph_Laplacian(K: dict, X: ArrayLike, v: ArrayLike):
  D1 = boundary_matrix(K, p = 1).tocsc().sorted_indices()
  fv = X @ np.array(v)
  fe = fv[K['edges']].max(axis=1)
  D1.data = np.sign(D1.data)*np.repeat(fe, 2)
  L = D1 @ D1.T
  return(L)

## Show num matvec's for Implicitly Restarted Lanczos 
# lanczos.sparse_lanczos(L, 29, 30, 1000, 1e-6)['n_operations']

# lanczos.sparse_lanczos(L, 10, 30, 1000, 1e-6)
L = weighted_graph_Laplacian(K, X, [1,0])

## Get threshold needed to obtain some percentage of the spectrum 
def trace_threshold(L, t=0.80):
  sum_ev2 = sum(L.diagonal())
  ev = primme.eigsh(L, k=L.shape[0]-1, ncv=L.shape[0], method="PRIMME_STEEPEST_DESCENT", return_eigenvectors=False, return_stats=False)
  thresh_ind = np.flatnonzero(np.cumsum(ev)/sum_ev2 > t)[0] # index needed to obtain 80% of the spectrum
  return thresh_ind

from pbsig.utility import timeout
from pbsig.precon import Jacobi, ssor, ShiftedJacobi

n,b,method=300,10,"PRIMME_Arnoldi"
K = { 'vertices': np.array(list(G.nodes)), 'edges': np.array(list(G.edges)) }
X = pht_preprocess_pc(np.array(list(nx.fruchterman_reingold_layout(G).values())))
L = weighted_graph_Laplacian(K, X, [1,0])
nev = trace_threshold(L, 0.80)

# https://www.cs.wm.edu/~andreas/software/doc/appendix.html#c.primme_params.correctionParams.projectors.SkewX

ew, stats = primme.eigsh(
  A=L, k=nev, method="PRIMME_Arnoldi", which="LM",
  #maxiter=nev,        # Max num. of *outer* iterations
  ncv=3,              # Max basis size / Lanczos vectors to keep in-memory; use minimum three for 3-term recurrence 
  maxBlockSize = 0,   # Max num of vectors added at every iteration. This affects the num. of shifts given to the preconditioner
  minRestartSize = 0, # Num. of approx. eigenvectors (Ritz vectors) kept during restart
  maxPrevRetain = 0,  # Num. approx eigenvectors kept *from previous iteration* in restart; the +k in GD+k
  #convtest=lambda eval, evec, resNorm: True,
  **default_opts
)


plt.plot(stats['hist']['numMatvecs'])
plt.plot(stats['hist']['nconv'])
plt.plot(np.cumsum(stats['hist']['resNorm']))

## Ranking p-faces of a (p+1)-simplex yields the orientations
# rank_combs([[0,1], [0,2], [1,2]], n=3, k=2) % 2


## Record stats 
STATS = {}
opts = default_opts.copy()
for n in [100, 500, 1000, 5000]:
  for b in [5, 10, 15, 20, 25]:
    for method in ["PRIMME_Arnoldi", "PRIMME_STEEPEST_DESCENT", "PRIMME_JDQR", "PRIMME_LOBPCG_OrthoBasis"]:
      G = nx.newman_watts_strogatz_graph(n=n, k=10, p=0.20)
      # G = nx.random_geometric_graph(n=n, radius=0.10, dim=3)
      # G = nx.erdos_renyi_graph(n=n, p=0.10)
      K = { 'vertices': np.array(list(G.nodes)), 'edges': np.array(list(G.edges)) }
      X = pht_preprocess_pc(np.array(list(nx.fruchterman_reingold_layout(G).values())))
      L = weighted_graph_Laplacian(K, X, [1,0])
      nev = trace_threshold(L, 0.80)
      opts |= dict(A=L, k=nev, ncv=b, method=method)
      # p_opts = dict(A=L, k=nev, ncv=nev+1, method="PRIMME_STEEPEST_DESCENT", which='LM', tol=1e-6, return_eigenvectors=False, return_stats=True)
      # primme.eigsh(**p_opts, OPinv=Jacobi(L))
      # primme.eigsh(**p_opts, OPinv=ssor(L))
      STATS[(n,b,method)] = timeout(primme.eigsh, kwargs=opts, timeout_duration=10, default=([], {}))
      print((n,b,method))
  

_, stats = primme.eigsh(A=L, k=29, ncv=5, method="PRIMME_Arnoldi", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_STEEPEST_DESCENT", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_GD", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_GD_Olsen_plusK", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_JDQR", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_JDQMR", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_JDQMR_ETol", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_JD_Olsen_plusK", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_LOBPCG_OrthoBasis", **opts)
_, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_LOBPCG_OrthoBasis_Window", **opts)


from scipy.sparse import diags, eye
from scipy.sparse.linalg import cg as conj_grad
class CG_OPinv(LinearOperator):
  def __init__(self, A, shift_f):
    self.shape, self.dtype = A.shape, A.dtype
    self.A = A 
    self.I = eye(A.shape[0]) # diags(np.repeat(1.0, self.shape[0]))
    self.D = diags(self.A.diagonal())
    self.n_calls = 0
    self.cg = conj_grad
    self.shift_f = shift_f
  def __update_cb__(self, xk: ArrayLike): 
    self.n_calls += 1
  def _matvec(self, x):
    if x.ndim == 1: return x * np.reciprocal(self.A.diagonal())
    y = np.copy(x)
    self.shifts = self.shift_f() #primme.get_eigsh_param('ShiftsForPreconditioner')
    assert all(abs(self.shifts) > 1e-13) 
    for i in range(x.shape[1]):
      # S = shifts[i]*self.I #S = np.array([s if s > 0 else 1 for s in S])
      # y[:,i] = self.cg(A=self.M - self.shifts[i]*self.I, b=x, tol=1e-6, callback=self.__update_cb__)[0]
      y[:,i] = self.cg(A=self.A - self.shifts[i]*self.I, b=x, tol=1e-6, callback=self.__update_cb__)[0]
    return y

from scipy.sparse.linalg import minres, gmres, qmr
x = np.random.uniform(size=L.shape[0], low=0.25, high=0.75)
# x /= np.linalg.norm(x)
b = L @ x

eye_unit = np.repeat(1, len(b))/np.linalg.norm(np.repeat(1, len(b)))
np.dot(b, eye_unit)
b - np.repeat(1, len(b))*(b @ np.repeat(1, len(b)))
x - minres(A=L, b=b, tol=1e-14)[0]
x - gmres(A=L, b=b, tol=1e-14)[0]
x - qmr(L, b, tol=1e-14)[0]

def print_stats(stats, pre):
  stat_keys = ['numOuterIterations', 'numRestarts', 'numMatvecs', 'numPreconds']
  print(str([(k, stats[k]) for k in stat_keys] + [("numPrecondIters", pre.n_calls)]))

## Using L=M itself as a preconditioner 
cg_pre = CG_OPinv(L, shift_f = lambda: primme.get_eigsh_param('ShiftsForPreconditioner'))
_, stats = primme.eigsh(L, k=29, ncv=30, which='LM', OPinv=cg_pre, method="PRIMME_GD_Olsen_plusK", return_stats=True, return_eigenvectors=False)
#print_stats(stats, cg_pre)

## Using D_L=M the diagonal as a preconditioner 
cg_pre = CG_OPinv(diags(L.diagonal()))
_, stats = primme.eigsh(L, k=29, ncv=30, which='LM', OPinv=cg_pre, method="PRIMME_GD_Olsen_plusK", **opts)
print_stats(stats, cg_pre)

## Using T=M as low-stretch spanning tree as a preconditioner 
from pbsig.lsst import low_stretch_st
M = low_stretch_st(G, method="akpw")
cg_pre = CG_OPinv(M)
_, stats = primme.eigsh(L, k=29, ncv=30, which='LM', OPinv=cg_pre, method="PRIMME_GD_Olsen_plusK", **opts)
print_stats(stats, cg_pre)



from pbsig.persistence import boundary_matrix
from scipy.sparse import diags
D1 = boundary_matrix(K, p = 1).tocsc().sorted_indices()
fv = np.random.uniform(size=len(K['vertices']), low=0.5, high=1.5)
fe = fv[K['edges']].max(axis=1)*1.1

D1.data = np.sign(D1.data)*np.repeat(fe, 2) # edge weighted case
LE = D1 @ D1.T
D1.data = np.sign(D1.data)
LE2 = D1 @ diags(fe**2) @ D1.T
max((LE.todense() - LE2.todense()).flatten())


D1.data = np.sign(D1.data)*fv[D1.indices] # edge weighted case
LV = D1 @ D1.T
D1.data = np.sign(D1.data)
# LV2 = diags(fv**2) @ D1 @ D1.T
# max((LV.todense() - LV2.todense()).flatten())
max(abs(np.ravel(((diags(fv) @ D1 @ D1.T @ diags(fv)).todense() - LV.todense()).flatten())))

import numpy as np 
from itertools import combinations
from gudhi import simplex_tree
from pbsig.utility import unrank_C2
nv = 50
st = simplex_tree.SimplexTree()
for i in range(nv): st.insert([i])
p = 0.40
P = np.random.uniform(size=int(nv*(nv-1)/2), low=0.0, high=1.0)
S = [st.insert(unrank_C2(cc,nv)) for cc in np.flatnonzero(P <= p)]
st.expansion(2)
K = dict(vertices=[], edges=[], triangles=[])
for s,fs in st.get_simplices():
  if len(s) == 1:
    K['vertices'].append(s[0])
  elif len(s) == 2: 
    K['edges'].append(s)
  if len(s) == 3 and np.random.uniform(0, 1) <= p:
    K['triangles'].append(s)

from pbsig.persistence import boundary_matrix
fv = np.random.uniform(size=len(K['vertices']), low=0.5, high=1.5)
fe = fv[K['edges']].max(axis=1)*1.1
ft = fv[K['triangles']].max(axis=1)*1.2

D2 = boundary_matrix(K, p = 2).tocsc().sorted_indices()
D2.data = np.sign(D2.data)*np.repeat(ft, 3) # triangle weighted case
LT = D2 @ D2.T
D2.data = np.sign(D2.data)
LT2 = D2 @ diags(ft**2) @ D2.T
max((LT.todense() - LT2.todense()).flatten())

from scipy.sparse import tril, diags
D2 = boundary_matrix(K, p = 2).tocsc().sorted_indices()
D2.data = np.sign(D2.data)*fe[D2.indices] # edge weighted case
LT = D2 @ D2.T
D2.data = np.sign(D2.data)
# LV2 = diags(fv**2) @ D1 @ D1.T
# max((LV.todense() - LV2.todense()).flatten())
max(abs(np.ravel(((diags(fe) @ D2 @ D2.T @ diags(fe)).todense() - LT.todense()).flatten())))



## Go around the sphere 
nd = 512
S1 = list(uniform_S1(nd))

from typing import *
def laplacian_dt(S1: Iterable, method: str, k: int, ncv: int, reuse_ev: bool = False):
  res = np.zeros(len(S1))
  opts['return_eigenvectors'] = reuse_ev
  for i, v in enumerate(S1):
    L = weighted_graph_Laplacian(K, v)
    if method == "IRL":
      stats = lanczos.sparse_lanczos(L, k, ncv, 1000, 1e-6)
      res[i] = stats['n_operations']
    elif reuse_ev == False:
      _, stats = primme.eigsh(L, k=k, ncv=ncv, method=method, **opts)
      res[i] = stats['numMatvecs']
    else:
      _, EV, stats = primme.eigsh(L, k=k, ncv=ncv, method=method, **opts) if i == 0 else primme.eigsh(L, k=k, ncv=ncv, v0=EV, method=method, **opts) 
      res[i] = stats['numMatvecs']
  return(res)

nmv_irl = laplacian_dt(S1, method="IRL", k=29, ncv=30)
nmv_jdqmr = laplacian_dt(S1, method="PRIMME_JDQMR", k=29, ncv=30)
nmv_jdqmr_ = laplacian_dt(S1, method="PRIMME_JDQMR", k=29, ncv=30, reuse_ev=True)
nmv_gd = laplacian_dt(S1, method="PRIMME_GD", k=29, ncv=30)
nmv_gd_ = laplacian_dt(S1, method="PRIMME_GD", k=29, ncv=30, reuse_ev=True)
nmv_gdo = laplacian_dt(S1, method="PRIMME_GD_Olsen_plusK", k=29, ncv=30)
nmv_gdo_ = laplacian_dt(S1, method="PRIMME_GD_Olsen_plusK", k=29, ncv=30, reuse_ev=True)
nmv_gdco = laplacian_dt(S1, method="PRIMME_STEEPEST_DESCENT", k=29, ncv=30)
nmv_gdco_ = laplacian_dt(S1, method="PRIMME_STEEPEST_DESCENT", k=29, ncv=30, reuse_ev=True)

import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(8,8), dpi=250)
ax = fig.gca()
ax.plot(np.cumsum(nmv_irl), label='IRL')
ax.plot(np.cumsum(nmv_jdqmr), label='GD', color='green')
ax.plot(np.cumsum(nmv_jdqmr_), label='GD+V', color='green', linestyle='dashed')
ax.plot(np.cumsum(nmv_gd), label='JDQMR', color='blue')
ax.plot(np.cumsum(nmv_gd_), label='JDQMR+V', color='blue', linestyle='dashed')
ax.plot(np.cumsum(nmv_gdo), label='GD (Olsen)', color='purple')
ax.plot(np.cumsum(nmv_gdo_), label='GD (Olsen) + V', color='purple', linestyle='dashed')
ax.plot(np.cumsum(nmv_gdco), label='GD (cheap Olsen)', color='red')
ax.plot(np.cumsum(nmv_gdco_), label='GD (cheap Olsen)+V', color='red', linestyle='dashed')
ax.legend()
ax.set_yscale('log')

plt.suptitle("Num. O(m) operations to obtain largest 30 EV's", y=0.950, fontsize=18)
plt.title(f"Directional Transform over Watts-Strogatz Graph w/ (n,m)={G.number_of_nodes(),G.number_of_edges()}")

plt.show()

(sum(nmv_irl)/512)/29.0
(sum(nmv_gdco)/512)/29.0





## Use previous eiegnvectors ?
# L = weighted_graph_Laplacian(K, [1, 0])
# opts['return_eigenvectors'] = True
# _, EV, stats = primme.eigsh(L, k=29, ncv=30, method="PRIMME_GD", **opts)
# _, EV, stats = primme.eigsh(L, k=29, ncv=30, v0 = EV, method="PRIMME_GD", **opts)













lanczos.sparse_lanczos(L, 29, 30, 1000, 1e-6)['n_operations']

primme.eigsh(L, k=1, ncv=15, which='LM', OPinv=cg_preconditioner, method="PRIMME_GD_Olsen_plusK", **opts)

# import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, coo_matrix
import numpy as np
import networkx as nx
from julia.api import Julia
jl = Julia(compiled_modules=False) # it's unclear why this is needed, but it is
from julia import Main
Main.using("Laplacians")
Main.using("LinearAlgebra")
Main.using("SparseArrays")
G = nx.connected_watts_strogatz_graph(10, k=3, p=0.20)
A = coo_matrix(nx.adjacency_matrix(G)).astype(float)
Main.I = np.array(A.row+1).astype(float)
Main.J = np.array(A.col+1).astype(float)
Main.V = A.data 
jl.eval("S = sparse(I,J,V)")
T = np.array(jl.eval("t = akpw(S)"))

# A = nx.adjacency_matrix(G).todense()
# A = [list(np.array(A[i,:]).astype(float).flatten()) for i in range(A.shape[0])]

nx.adjacency_matrix(G)

# A = jl.eval(f"A = empty_graph({ G.number_of_nodes() })")
# for u,v in combinations(range(G.number_of_nodes()), 2):
#   A[u][v] = A[v][u] = G.edges[u,v]['weight'] if (u,v) in G.edges else 0.0 
# A = jl.eval("A = dropzeros(A)")
# A = jl.eval("A = uniformWeight(A)")
jl.eval("I = [1,2,3]; J = [1,2,3]; V = [1,1,1];")
jl.eval("S = sparse(I,J,V)")
jl.eval("t = akpw(S)")

G = jl.eval("G = pure_random_graph(10)")
G = jl.eval("G = dropzeros(G)")
np.array(jl.eval("t = akpw(G)"))
# A = jl.eval("zeros(3,3)")
Main.G = G
np.array(jl.eval("t = akpw(G)"))


# mst = np.array(jl.eval("mst = kruskal(a)"))
st = np.array(jl.eval("t = akpw(G)"))
st = np.array(jl.eval("t = randishPrim(a)"))
st = np.array(jl.eval("t = randishKruskal(a)"))
st.sum()
np.array(jl.eval("comp_stretches(t,a)")).sum()

import networkx as nx
G = nx.from_numpy_array(np.array(A))
T = nx.Graph()
T.add_nodes_from(range(st.shape[0]))
T = nx.from_numpy_matrix(st)
assert nx.is_connected(T) and nx.is_tree(T)
# DW = nx.floyd_warshall_numpy(T)
from pbsig.utility import pairwise
nv = T.number_of_nodes()
stretch = 0.0
for u,v in G.edges:
  e_weight = G.edges[u,v]['weight']
  e_path = np.array(nx.bellman_ford_path(T, u, v))
  dist_uv = sum([T.edges[i,j]['weight'] for (i,j) in pairwise(e_path)])
  stretch += dist_uv / (1.0 / e_weight)
stretch /= G.number_of_edges()

# nx.draw(T)


(1.0/st).sum()
# f = pcgLapSolver(a,t)
# tree = akpw(a);

st = nx.minimum_spanning_tree(G)
M = nx.adjacency_matrix(st)
CG_OPinv(M)



## Laplacian matrices are normal matrices! 
A = nx.adjacency_matrix(G)
D = np.diag(np.array([G.degree(i) for i in G.nodes]))
L = D - A
all(np.ravel(((L @ L.T) == (L.T @ L)).flatten()))


# x = np.random.uniform(size=L.shape[0], low=-0.50, high=0.50)
# shift = 2.0862413029866818
# I = np.eye(len(x))
# b = (L - shift*I) @ x
# max(abs(cg(A=L - shift * I, b=b, tol=1e-12)[0] - x))
## PRIMME_GD := Generalized Davidson.
## PRIMME_JDQR := Jacobi-Davidson with fixed number of inner steps.
## PRIMME_GD_plusK := GD with locally optimal restarting.
## PRIMME_JDQMR := Jacobi-Davidson with adaptive stopping criterion for inner Quasi Minimum Residual (QMR).
## (min matvec's) PRIMME_GD_Olsen_plusK := GD+k and the cheap Olsen’s Method. 