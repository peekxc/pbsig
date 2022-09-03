import numpy as np 
from typing import *

from pbsig.utility import rank_C2

K = {
  'vertices' : [0,1,2,3,4,5,6,7,8],
  'edges' : [[0,1],[0,2],[0,7],[1,2],[1,8],[2,3],[2,4],[3,4],[3,5],[3,6],[4,6],[6,7],[7,8]],
  'triangles' : [[0,1,2],[2,3,4]]
}

## Random vertex height values
fv = np.random.uniform(size=len(K['vertices']), low=0, high=1)
fe = np.array([max(fv[u],fv[v]) for u,v in K['edges']])

## Replace elementary chains
from pbsig.persistence import boundary_matrix
D = boundary_matrix(K, p = 1)
D.data = np.sign(D.data)*np.repeat(fe, 2)

## O(n) matrix-vec multiplication example 
x = np.random.uniform(size=D.shape[0])[:,np.newaxis] # V
y = D.T @ x # E 
z = D @ y # V

r = np.zeros(len(y))
for cc, (i,j) in enumerate(K['edges']):
  r[cc] = fe[cc]*x[i] - fe[cc]*x[j]

rz = np.zeros(D.shape[0])
for cc, (i,j) in enumerate(K['edges']):
  rz[i] += fe[cc]*r[cc]
  rz[j] -= fe[cc]*r[cc]

## Graph laplacian formulation 
# W = [max(f_sigma[e]) for e in K['edges']]
# y = np.array([w_uv*(x[u] - x[v])**2 for (u,v), w_uv in zip(K['edges'], W)])

#Z @ x


## Mat-vec on (n x m) matrix with n < m takes O(m) time for boundary matrices! 
## w/ O(m + n) storage, 
# z.flatten() - rz
Z = (D @ D.T)
abs(Z.diagonal()) - abs(Z - np.diag(Z.diagonal())).sum(axis=0) # diagonally dominant


from scipy.sparse.linalg import LinearOperator, eigsh
from numpy.typing import ArrayLike
from typing import * 

def boundary1_matvec(shape: Tuple, E: List, fe: ArrayLike):
  n, m = shape
  r, rz = np.zeros(m), np.zeros(n)
  def _mat_vec(x: ArrayLike):
    r.fill(0)
    rz.fill(0)
    for cc, (i,j) in enumerate(E):
      r[cc] = fe[cc]*x[i] + -fe[cc]*x[j]
    for cc, (i,j) in enumerate(E):
      rz[i] += fe[cc]*r[cc]
      rz[j] -= fe[cc]*r[cc]
    return(rz)
  return(_mat_vec)

matvec = boundary1_matvec(D.shape, K['edges'], fe)
A = LinearOperator((D.shape[0], D.shape[0]), matvec)

((D @ D.T) @ x) - A(x)

ev_D = eigsh(D @ D.T, k=min(D.shape)-1, return_eigenvectors=False)
ev_A = eigsh(A, k=min(D.shape)-1, return_eigenvectors=False)


eigsh(A, k=min(D.shape)-1, return_eigenvectors=False)

def boundary1_lowerstar_lex(shape: Tuple, E: List, fv: ArrayLike):
  nv, ne = shape
  r, rz = np.zeros(ne), np.zeros(nv)
  def _mat_vec(x: ArrayLike): # x ~ O(V)
    r.fill(0) # r = Ax ~ O(E)
    rz.fill(0)# rz = Ar ~ O(V)
    for cc, (i,j) in enumerate(E):
      ew = fv[i] if fv[i] >= fv[j] else fv[j]
      r[cc] = ew*x[i] - ew*x[j]
    for cc, (i,j) in enumerate(E):
      ew = fv[i] if fv[i] >= fv[j] else fv[j]
      rz[i] += ew*r[cc]
      rz[j] -= ew*r[cc]
    return(rz)
  return(_mat_vec)

matvec = boundary1_lowerstar_lex(D.shape, K['edges'], fv)
A = LinearOperator((D.shape[0], D.shape[0]), matvec)
((D @ D.T) @ x) - A(x)

ev_D = eigsh(D @ D.T, k=min(D.shape)-1, return_eigenvectors=False)
ev_A = eigsh(A, k=min(D.shape)-1, return_eigenvectors=False)


## For general sparse egdes, need  
# rank_C2(i,j, n=len(K['vertices']))
# np.searchsorted(e_ind, [i,j,l]) ...
# O(E) time / O(V^2) memory version
def UL2_matvec_lower_star_lex(shape: Tuple, T: List, fv: ArrayLike, E: List = None):
  from pbsig.utility import rank_combs
  from math import comb
  ne, nt = shape
  N = len(fv)
  r, rz = np.zeros(nt), np.zeros(comb(N, 2))
  e_ind = rank_combs(E, k=2, n=N)
  assert len(e_ind) == ne, ""
  X = np.zeros(comb(N, 2))
  # D.T @ x
  def _mat_vec(x: ArrayLike): # E 
    X[e_ind] = x.flatten()
    r.fill(0)
    rz.fill(0)
    for cc, (i,j,k) in enumerate(T):
      I,J,K = rank_combs([[i,j], [i,k], [j,k]], k=2, n=N)
      tv = max([fv[i], fv[j], fv[k]])
      r[cc] = tv*X[I] - tv*X[J] + tv*X[K]
    for cc, (i,j,k) in enumerate(T):
      tv = max([fv[i], fv[j], fv[k]])
      rz[rank_C2(i,j,N)] += tv*r[cc]
      rz[rank_C2(i,k,N)] -= tv*r[cc]
      rz[rank_C2(j,k,N)] += tv*r[cc]
    return(rz[e_ind])
  return(_mat_vec)



D = boundary_matrix(K, p = 2)
fv = np.random.uniform(size=len(K['vertices']), low=0, high=1)
fe = np.array([max(fv[e]) for e in K['edges']])
ft = np.array([max(fv[t]) for t in K['triangles']])
D.data = np.sign(D.data)*np.repeat(ft, 3)

matvec = UL2_matvec_lower_star_lex(D.shape, K['triangles'], fv, K['edges'])
A = LinearOperator((D.shape[0], D.shape[0]), matvec)

#x = np.random.uniform(size=D.shape[0])[:,np.newaxis] # V
((D @ D.T) @ x) - A(x)

ev_D = eigsh(D @ D.T, k=min(D.shape), return_eigenvectors=False)
ev_A = eigsh(A, k=min(D.shape), return_eigenvectors=False)


## Custom 
from pbsig.linalg import lanczos
E = np.array(K['edges'], dtype=np.float32)
x = np.random.uniform(size=len(K['vertices'])).astype(np.float32)[:,np.newaxis]

z1 = ((D @ D.T) @ x).flatten()
z2 = lanczos.UL1_LS_matvec(x, fv, E[:,0], E[:,1])
abs(z1-z2)


lanczos.UL1_LS_lanczos(fv, E[:,0], E[:,1], 1, 9)