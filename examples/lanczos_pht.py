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

import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse
from scipy.sparse import random as random_sp
import primme


A = random_sp(300,300,density=0.15)
M = A @ A.T


lanczos.sparse_lanczos(M, 10, 11, 150, 1e-6) # M, nev, ncv, max_iter, tol

ew, ev, res = primme.eigsh(M, k=10, maxiter=150, tol=1e-6, which='LM', return_stats=True) # OPinv
print(res['numMatvecs'])
ew2, ev2, res2 = primme.eigsh(M, k=10, tol=1e-6, v0=ev2, which='LM', return_stats=True)
print(res2['numMatvecs'])
# Conjecture: does computing the p-th Betti number of a complex with n = |K^p| have ~Omega(nr) complexity, for p >= 1?

# OPinv (N x N matrix, array, sparse matrix, or LinearOperator, optional) 
# Preconditioner to accelerate the convergence. Usually it is an approximation of the inverse of (A - sigma*M)

# lock (N x i, ndarray, optional) – Seek the eigenvectors orthogonal to these ones. 
# The provided vectors should be orthonormal. Useful to avoid converging to previously computed solutions.

from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigsh, minres, cg, LinearOperator

eigsh(M, k=1, sigma=500.0)[0]

class CG_OPinv(LinearOperator):
  def __init__(self, shift, M):
    self.shape = M.shape
    self.dtype = M.dtype
    self.shift = shift
    self.M = M 
  def _matvec(self, x):
    return cg(M - self.shift*diags(np.repeat(1.0, self.shape[0])), x)[0]

x = ev[:,0]
assert np.allclose(M @ x, ew[0] * x)  ## indeed it is an eigenvector
assert np.allclose(x, cg(M, M @ x)[0]) ## asserts CG solves the system A x = b when b = \lambda x 
cg(M, x)

ew
eigsh(M, k=1, sigma=50.0, OPinv=CG_OPinv(50.0, M))[0]

eigsh(M, k=1, sigma=70.0, OPinv=CG_OPinv(70.0, M))[0]


eigsh(M, k=1, sigma=70.0, OPinv=op)[0]
eigsh(M, k=1, sigma=50.0)[0]

eigsh(M - 50.0*diags(np.repeat(1.0, M.shape[0])), k=1, which='LM', mode='normal')[0]

50.0 + 1/484.32684885

primme.eigsh(M, k=10, maxiter=150, tol=1e-6, which='LM', return_stats=True)



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





import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse
import primme
A = scipy.sparse.spdiags(np.asarray(range(100), dtype=np.float32)+1, [0], 100, 100)
def P(x):
   # The scipy.sparse.linalg.LinearOperator constructor may call this function giving a vector
   # as input; detect that case and return whatever
   if x.ndim == 1:
      return x / A.diagonal()
   shifts = primme.get_eigsh_param('ShiftsForPreconditioner')
   print(len(shifts))
   y = np.copy(x)
   for i in range(x.shape[1]): 
    S = A.diagonal() - shifts[i]
    S = np.array([s if s > 0 else 1 for s in S])
    y[:,i] = x[:,i] / S # since D^{-1} == 1/d_ii for diagonal matrix
   return y
Pop = scipy.sparse.linalg.LinearOperator(A.shape, matvec=P, matmat=P)



ew, stats = primme.eigsh(
  A, 10, OPinv=Pop,
  maxiter=15000, maxBlockSize=20, tol=np.finfo(np.float32).eps,
  #maxBlockSize=100, tol=1e-2 
  which='LM', return_eigenvectors=False, 
  return_stats=True, raise_for_unconverged=False, return_history=False
)


def eigsh_block_shifted(A, k: int, b: int = 10, **kwargs):
  ni = int(np.ceil(k/b))
  f_args = dict(tol=np.finfo(np.float32).eps, which='CGT', return_eigenvectors=True, raise_for_unconverged=False, return_history=False)
  f_args = f_args | kwargs
  evals = np.zeros(ni*b)
  for i in range(ni):
    if i == 0:
      ew, ev = primme.eigsh(A, k=b, **f_args)
    else: 
      #f_args['which'] = min(ew)
      f_args['sigma'] = min(ew)
      ew, ev = primme.eigsh(A, k=b, lock=ev, **f_args)
    print(ew)
    evals[(i*b):((i+1)*b)] = ew
  return(evals)

## How to avoid getting eigenvalues already sought? Use CGT! 

eigsh_block_shifted(A, k=10, b=3)
eigsh_block_shifted(A, k=10, b=3, OPinv=Pop)
# OPinv=Pop

## Preconditioned CG testing 
from scipy.sparse.linalg import cg

n_iter = 0
def inc_cb(xk):
  global n_iter 
  n_iter += 1
x = np.random.uniform(size=A.shape[0], low=-0.50, high=0.50)
b = A @ x

## Test regular CG 
xs, _ = cg(A=A, b=b, callback=inc_cb, tol=1e-12)
assert max(abs(xs - x)) <= 1e-11

## PCG with exact inverse
from scipy.sparse import diags
M = diags(1/A.diagonal())
xs, _ = cg(A=A, b=b, M=M, callback=inc_cb, tol=1e-12)
assert max(abs(xs - x)) <= 1e-11


f_args = dict(tol=np.finfo(np.float32).eps, which='LM', return_eigenvectors=False, raise_for_unconverged=False, return_stats=True, return_history=False)

primme.eigsh(A, k=30, **f_args)
primme.eigsh(A, k=30, OPinv=M, **f_args)