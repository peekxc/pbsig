from typing import * 
import numpy as np 
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator

from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1
from pbsig.datasets import mpeg7
from pbsig.pht import pht_preprocess_pc

class UpLaplacianVE1_ls(LinearOperator):
  def __init__(self, K: dict, fv: ArrayLike):
    self.nv = len(fv)
    self.fv = fv
    self.fe = lambda i,j: max(self.fv[i],self.fv[j])
    self.z: ArrayLike = np.zeros(self.nv)
    self.Df: ArrayLike= np.zeros(self.nv)
    for cc, (i,j) in enumerate(K['edges']):
      self.Df[i] += self.fe(i,j)**2
      self.Df[j] += self.fe(i,j)**2
    self.shape = (int(self.nv), int(self.nv))
    self.dtype = np.dtype(np.float64)

  def _matvec(self, x: ArrayLike):
    self.z.fill(0.0)
    for i in range(self.nv):
      self.z[i] += x[i]*self.Df[i]*self.fv[i]**2
    for cc, (i,j) in enumerate(K['edges']):
      self.z[i] -= x[j] * self.fe(i,j)**2 * self.fv[i] * self.fv[j]
      self.z[j] -= x[i] * self.fe(i,j)**2 * self.fv[i] * self.fv[j]
    return self.z 

dataset = mpeg7(simplify=150)
X = dataset[('turtle',1)]
X = pht_preprocess_pc(X)
K = cycle_graph(X)

## verify eigenvalues 
fv = X @ np.array([1,0])
fe = fv[K['edges']].max(axis=1)
# DV, DE = boundary_matrix(K, p = 1).tocsc(), boundary_matrix(K, p = 1).tocsc()
# DE.sort_indices()
# DE.data = np.sign(DE.data)*np.repeat(fe, 2)

F = list(rotate_S1(X, 132, include_direction=False))
I = np.array(K['edges'])[:,0].astype(int)
J = np.array(K['edges'])[:,1].astype(int)

np.random.seed(1234)
v0 = np.random.uniform(size=len(fv), low=-0.50, high=0.50)
#theta = np.linspace(0, 2*np.pi, 132, endpoint=False)
theta = np.zeros(10)
lanczos.UL0_VELS_PHT_2D(X, theta, I, J, 30, 40, 0, 1e-5, v0, 0, 0, 0, 0)
# 392 ops, 10 restarts for 10 zero-theta's w/ 0 maxiter

# Even all the zero vector takes large amount. Why? 
# How to re-use any of the computation?
# 10585
# 9960

## JD? 
import pysparse
from pysparse.sparse import spmatrix
from pysparse.itsolvers import krylov
from pysparse.eig import jdsym
from pysparse.precon import precon

A = spmatrix.ll_mat_from_mtx('edge6x3x5_A.mtx')
M = spmatrix.ll_mat_from_mtx('edge6x3x5_B.mtx')
tau = 25.0

Atau = A.copy()
Atau.shift(-tau, M)
K = precon.jacobi(Atau)

A = A.to_sss(); M = M.to_sss()
k_conv, lmbd, Q, it  = jdsym.jdsym(A, M, K, 5, tau,
                                   1e-10, 150, krylov.qmrs,
                                   jmin=5, jmax=10, clvl=1, strategy=1)












from pbsig.linalg import lanczos
n_ops = 0
for i, fv in enumerate(F): 
  # v0 = np.random.uniform(size=len(fv), low=-0.50, high=0.50)
  res = lanczos.UL0_VELS_lanczos(fv, I, J, 30, 40, 100, 1e-5, v0, 0, 0, 0, 0)
  n_ops += res['n_operations']
print(n_ops) # is identical! 
# 9302

from scipy.sparse.linalg import aslinearoperator, eigsh
n_ops = 0
v0 = np.random.uniform(size=X.shape[0], low=-0.50, high=0.50)
for i, fv in enumerate(F): 
  res = lanczos.UL0_VELS_lanczos(fv, I, J, 30, 40, 100, 1e-5, v0, 0, 0, 0, 0)
  n_ops += res['n_operations']
  L_op = aslinearoperator(UpLaplacianVE1_ls(K, fv))
  _, v0 = eigsh(L_op, k=1, return_eigenvectors=True)
print(n_ops)


## UpLaplacian0 - Vertex/Edge Lower Star
U = np.zeros(shape=(132, 140))
for i, fv in enumerate(F): 
  res = lanczos.UL0_VELS_lanczos(fv, I, J, 140, 141, 1000, 1e-12)
  U[i,:] = res['ritz_values'][:140]

for j in range(U.shape[1]):
  plt.plot(U[:,j])
plt.plot(U.sum(axis=1))