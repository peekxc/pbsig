## SUMMARY OF FINDINGS: Estimating dim(nullspace(L)) is actually just as hard (if not harder!) as estimating the full spectrum of Laplacians!
import numpy as np 
from scipy.sparse import diags, eye 
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh
from scipy.sparse.csgraph import laplacian

import networkx as nx 
import primme

G = nx.connected_watts_strogatz_graph(n=100, k=5, p=0.10)
W = diags(np.random.uniform(size=G.number_of_nodes(), low=0.0, high=1.0))
L = W @ laplacian(nx.adjacency_matrix(G)) @ W

#D1 = boundary_matrix(graph2complex(G), p=1)
#Z = (W @ D1 @ D1.T @ W) - L

D = L.diagonal()
VD = np.zeros(L.shape[0])
for i,j in G.edges():
  VD[i] += D[j]
  VD[j] += D[i]
np.max(np.sqrt(VD))

np.sqrt(2)*np.sqrt(max(D**2 + VD)) ## should be an upper bound on Laplacian spectrla radius 
max(eigsh(L)[0])


class LinearOperatorWithStats(LinearOperator):
  def __init__(self, A):
    self.A = A
    self.dtype = A.dtype
    self.shape = A.shape
    self.n_calls = 0

  def _matvec(self, x):
    self.n_calls += 1
    return self.A @ x

  def _matmat(self, X):
    self.n_calls += X.shape[1]
    return self.A @ X

from scipy.linalg import qr
normalize = lambda x: x / np.linalg.norm(x)
v0 = normalize(np.repeat(1, L.shape[0]))[:,np.newaxis]
# V = np.apply_along_axis(normalize, 0, np.random.uniform(size=(L.shape[0], 2)))
Q, _ = qr(np.c_[v0, np.random.uniform(size=(L.shape[0], 2))], mode='economic')
# check orthogonality Q.T @ Q

ew, stats = primme.eigsh(L, k = 1, ncv=3, which='SM', tol=1e-6, v0=Q, method="PRIMME_Arnoldi", return_stats=True, return_eigenvectors=False)
stats

## Use various norms computable in linear-time to bound spectral norm 
n = L.shape[0]
L_max = np.max(abs(L.data))
L_fro = np.sqrt(np.sum(L.data**2))
L_one = np.max(abs(L).sum(axis=0))
L_inf = np.max(abs(L).sum(axis=1))
sigma = np.min([n*L_max, np.sqrt(n)*L_inf, np.sqrt(n)*L_one, np.sqrt(L_one*L_inf)])


sI = sigma*eye(L.shape[0])
ew, stats = primme.eigsh(L-sI, k = 1, ncv=3, which='LM', tol=1e-6, v0=Q, method="PRIMME_Arnoldi", return_stats=True, return_eigenvectors=False)
stats

#ew, stats = primme.eigsh(L-sI, k = 1, ncv=3, which='LM', tol=1e-6, v0=Q, method="PRIMME_JDQMR", return_stats=True, return_eigenvectors=False)
ew, stats = primme.eigsh(L-sI, k = 1, ncv=10, which='LM', tol=1e-6, v0=Q, method="PRIMME_DYNAMIC", return_stats=True, return_eigenvectors=False)
stats

# https://docs.scipy.org/doc/scipy/tutorial/arpack.html

LO = LinearOperatorWithStats(L)
w = eigsh(LO, k = 1, which='SM', tol=1e-6, v0=v0, return_eigenvectors=False)
LO.n_calls

## Let's get the largest eigenvalue! 
LO = LinearOperatorWithStats(L-sI)
v = eigsh(LO, k = 1, which='SM', tol=1e-6, v0=v0, return_eigenvectors=False)
LO.n_calls
v + sigma ## recovers the max lambda value
ew_max = max(np.linalg.eigvalsh(L.todense()))

## Let's get the smallest eigenvalue! 
LO = LinearOperatorWithStats(L-sI)
v = eigsh(LO, k = 1, which='LM', tol=1e-6, v0=v0, return_eigenvectors=False)
LO.n_calls # 890 
ew_min = min(np.linalg.eigvalsh(L.todense()))
v + sigma ## recovers the min lambda value

## Let's use the explicit shifting mechanism
LO = LinearOperatorWithStats(L)
v = eigsh(LO, k = 1, which='SM', tol=1e-6, v0=v0, sigma=sigma, return_eigenvectors=False, maxiter=15000)
LO.n_calls


primme.eigsh(L, k=1, which='SM', tol=1e-6, return_eigenvectors=False, method="PRIMME_Arnoldi", return_stats=True)

x = np.random.uniform(size=L.shape[0], low=-1.0, high=1.0)
x @ L @ x

np.linalg.eigvalsh((L-sI).todense())

assert all(np.linalg.eigvalsh(L.todense()) >= 0)

plt.plot(eigsh(L, k=L.shape[0]-1)[0])


LO = LinearOperatorWithStats(L)
ew = eigsh(LO, k = L.shape[0]-1, which='LM', tol=1e-6, return_eigenvectors=False)
LO.n_calls


ew, stats = primme.eigsh(L, ncv=L.shape[0], k=L.shape[0]-1, which='LM', v0=v0, tol=1e-6, return_eigenvectors=False, method="PRIMME_Arnoldi", return_stats=True)


ew, stats = primme.eigsh(L, ncv=L.shape[0], k=L.shape[0]-1, which='LM', v0=v0, tol=1e-6, return_eigenvectors=False, method="PRIMME_STEEPEST_DESCENT", return_stats=True)