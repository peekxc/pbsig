import numpy as np 
import networkx as nx 
from splex import * 
from itertools import * 
from pbsig.linalg import up_laplacian
from pbsig.vis import plot_complex


from scipy.sparse import diags
np.random.seed(1234)
G = nx.random_geometric_graph(100, 0.30)
S = SimplicialComplex(chain(G.nodes, G.edges))
X = np.array([d['pos'] for n, d in G.nodes(data=True)])
ew = np.array([int(20*np.linalg.norm(X[i,:]-X[j,:]))+1 for i,j in S.faces(1)])

from pbsig.lsst import sparsify
from pbsig.linalg import adjacency_matrix
A = adjacency_matrix(S, weights=ew)
L = diags(A.sum(axis=0)) - A

from scipy.sparse import diags
n = A.shape[0]
for epsilon in np.linspace(1.0, 1/np.sqrt(n), 20):
  #epsilon = 0.9
  A_sparse = sparsify(A, epsilon=epsilon, ensure_connected=True, max_tries = 200, exclude_singletons=False)
  L_sparse = diags(A_sparse.sum(axis=0)) - A_sparse

  ## Check the supposed guarentees 
  N = 1500 # number of queries 
  is_satisfied = np.zeros(N, dtype=bool)
  for i in range(N):
    x = np.random.uniform(size=L.shape[0], low=-1.0, high=1.0)
    lb = (1 - epsilon) * x.T @ L @ x
    ub = (1 + epsilon) * x.T @ L @ x
    sq = x.T @ L_sparse @ x
    is_satisfied[i] = bool((lb <= sq) and (sq <= ub))
  r = int(L_sparse.nnz/2)/int(L.nnz/2)
  print(f"Sparsification for eps={epsilon:.2}: {r:.2}")
  assert sum(is_satisfied) >= len(is_satisfied)/2, "This should hold in expectation"
  if r == 1.0: 
    break

## Given a sparsified adjacency, extract the subcomplex
A_sparse = sparsify(A, epsilon=1.0, ensure_connected=True, max_tries = 200, exclude_singletons=False)









## Compute effective resistance
D = boundary_matrix(S, p=1).tocsc()
P = np.linalg.pinv(L.todense())
ER = np.array([float(D[:,[j]].T @ P @ D[:,[j]]) for j in range(S.shape[1])])
  
## Show embedding 
X = np.array([d['pos'] for n, d in G.nodes(data=True)])
plot_complex(S, X)

## Verify claims 
n,m = S.shape[:2]
sum(ER) <= (n-1)/m                 # False

## Generate random geometric graph
from scipy.sparse import diags
np.random.seed(1234)
G = nx.random_geometric_graph(100, 0.30)
S = SimplicialComplex(chain(G.nodes, G.edges))
X = np.array([d['pos'] for n, d in G.nodes(data=True)])
# ew = np.random.choice(range(20), size=S.shape[1]) # integer edge weights 
ew = np.array([int(20*np.linalg.norm(X[i,:]-X[j,:]))+1 for i,j in S.faces(1)])

## Sparsification algorithm 
k = int(np.floor(np.log2(n)))
sparsified = [None]*k
for i in range(k):

  ## Save information 
  D = boundary_matrix(S, p=1).tocsc()
  L = D @ diags(ew) @ D.T
  sparsified[i] = (S, L, ew)
  
  ## Compute effective resistence
  P = np.linalg.pinv(L.todense())
  ER = np.array([float(D[:,[j]].T @ P @ D[:,[j]]) for j in range(S.shape[1])])
  
  from pygsp.utils import resistance_distance
  len(resistance_distance(L).toarray())
  ## Sample 
  p = (ER * ew)/sum(ER * ew)
  n,m = S.shape[:2]
  assert sum(ew*ER <= (2*n / m)) >= len(ER)/2, "Need at least half" # True 
  # # q = int(9*n*np.log2(n)/(1/np.sqrt(n))**2)
  ## See: https://github.com/epfl-lts2/pygsp/blob/a7c627e104fd96495c780814ed81c2d9862723d7/pygsp/reduction.py
  # np.random.choice(range(S.shape[1]), size=, p=p, replace=True)
  # ind = np.flatnonzero(ew*ER <= (2*n / m ))
  # I = np.zeros(m)
  # I[ind] = np.random.choice([0,1], size=len(ind))
  S = SimplicialComplex(chain(S.faces(0), compress(S.faces(1), I)))


# from pygsp.reduction import graph_sparsify
# graph_sparsify(L, epsilon=0.99)
