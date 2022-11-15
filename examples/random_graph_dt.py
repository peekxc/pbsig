from typing import * 
import numpy as np 
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator
import networkx as nx 

from pbsig.persistence import boundary_matrix
from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1
from pbsig.datasets import mpeg7
from pbsig.pht import pht_preprocess_pc

import primme
## PRIMME_GD := Generalized Davidson.
## PRIMME_JDQR := Jacobi-Davidson with fixed number of inner steps.
## PRIMME_GD_plusK := GD with locally optimal restarting.
## PRIMME_JDQMR := Jacobi-Davidson with adaptive stopping criterion for inner Quasi Minimum Residual (QMR).
## (min matvec's) PRIMME_GD_Olsen_plusK := GD+k and the cheap Olsen’s Method. 
opts = dict(tol=1e-6, printLevel=0, return_eigenvectors=False, return_stats=True)

G = nx.connected_watts_strogatz_graph(130, k=7, p=0.20)
K = { 'vertices': np.array(list(G.nodes)), 'edges': np.array(list(G.edges)) }
X = pht_preprocess_pc(np.array(list(nx.fruchterman_reingold_layout(G).values())))

fig = plt.figure(figsize=(5,5))
nx.draw(G, pos=X, ax=plt.axes(), node_size=15)

## Function to generate matrices
def weighted_graph_Laplacian(K: dict, v: ArrayLike):
  D1 = boundary_matrix(K, p = 1).tocsc().sorted_indices()
  fv = X @ np.array(v)
  fe = fv[K['edges']].max(axis=1)
  D1.data = np.sign(D1.data)*np.repeat(fe, 2)
  L = D1 @ D1.T
  return(L)

## Show num matvec's for Implicitly Restarted Lanczos 
# lanczos.sparse_lanczos(L, 29, 30, 1000, 1e-6)['n_operations']

# lanczos.sparse_lanczos(L, 10, 30, 1000, 1e-6)
L = weighted_graph_Laplacian(K, [1,0])
stats = lanczos.sparse_lanczos(L, 29, 30, 1000, 1e-6)
_, stats = primme.eigsh(L, k=29, ncv=30, tol=1e-6, return_stats=True, return_eigenvectors=False, method="PRIMME_RQI")
_, stats = primme.eigsh(L, k=29, ncv=30, tol=1e-6, return_stats=True, return_eigenvectors=False, method="PRIMME_STEEPEST_DESCENT")
_, stats = primme.eigsh(L, k=29, ncv=30, tol=1e-6, return_stats=True, return_eigenvectors=False, method="PRIMME_Arnoldi")
_, stats = primme.eigsh(L, k=29, ncv=30, tol=1e-6, return_stats=True, return_eigenvectors=False, method="PRIMME_GD_Olsen_plusK")

class CG_OPinv(LinearOperator):
  def __init__(self, M):
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg as conj_grad
    self.shape, self.dtype = M.shape, M.dtype
    self.M = M 
    self.I = diags(np.repeat(1.0, self.shape[0]))
    self.n_calls = 0
    self.cg = conj_grad
  def __update_cb__(self, xk: ArrayLike): 
    self.n_calls += 1
  def _matvec(self, x):
    if x.ndim == 1: 
      return x / self.M.diagonal()
    shifts = primme.get_eigsh_param('ShiftsForPreconditioner')
    y = np.copy(x)
    self.shifts = shifts
    self.y = y
    for i in range(x.shape[1]):
      # S = shifts[i]*self.I #S = np.array([s if s > 0 else 1 for s in S])
      y[:,i] = cg(A=self.M - shifts[i]*self.I, b=x, tol=1e-6, callback=self.__update_cb__)[0]
    return y

cg_pre = CG_OPinv(L)
cg_pre = CG_OPinv(diags(L.diagonal()))

_, stats = primme.eigsh(L, k=29, ncv=30, which='LM', OPinv=cg_pre, method="PRIMME_GD_Olsen_plusK", **opts)



cg_pre.n_calls






lanczos.sparse_lanczos(L, 29, 30, 1000, 1e-6)['n_operations']

primme.eigsh(L, k=1, ncv=15, which='LM', OPinv=cg_preconditioner, method="PRIMME_GD_Olsen_plusK", **opts)

# import matplotlib.pyplot as plt
import numpy as np
from julia.api import Julia
jl = Julia(compiled_modules=False) # it's unclear why this is needed, but it is
from julia import Main
Main.using("Laplacians")
Main.using("LinearAlgebra")
Main.using("SparseArrays")
# Main.G = 
from itertools import combinations
import networkx as nx
# G = nx.from_numpy_array(L)

# A = jl.eval(f"A = empty_graph({ G.number_of_nodes() })")
# for u,v in combinations(range(G.number_of_nodes()), 2):
#   A[u][v] = A[v][u] = G.edges[u,v]['weight'] if (u,v) in G.edges else 0.0 
# A = jl.eval("A = dropzeros(A)")
# A = jl.eval("A = uniformWeight(A)")
jl.eval("I = [1,2,3]; J = [1,2,3]; V = [1,1,1];")
jl.eval("S = sparse(I,J,V)")


G = jl.eval("G = pure_random_graph(10)")
np.array(jl.eval("t = akpw(G)"))
# A = jl.eval("zeros(3,3)")
Main.G = A

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
