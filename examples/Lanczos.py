from typing import *
import numpy as np 
import networkx as nx
from pbsig.utility import rank_C2
from pbsig.persistence import boundary_matrix
import matplotlib.pyplot as plt 

K = {
  'vertices' : [0,1,2,3,4,5,6,7,8],
  'edges' : [[0,1],[0,2],[0,7],[1,2],[1,8],[2,3],[2,4],[3,4],[3,5],[3,6],[4,6],[6,7],[7,8]],
  'triangles' : [[0,1,2],[2,3,4]]
}
G = nx.Graph()
G.add_nodes_from(K['vertices'])
G.add_edges_from(K['edges'])

## Random vertex+edge height values
fv = np.random.uniform(size=len(K['vertices']), low=0, high=1)
fe = np.array([max(fv[u],fv[v]) for u,v in K['edges']])

## Replace elementary chains w/ edge-values
DE = boundary_matrix(K, p = 1).tocsc()
DE.sort_indices()
DE.data = np.sign(DE.data)*np.repeat(fe, 2)

## Look at the values of the matrix + its Laplacian 
plt.imshow(abs(DE.A),interpolation='none',cmap='binary')
plt.imshow(abs((DE @ DE.T).A), interpolation='none',cmap='binary')

## Check that it matches the explicit form of the matrix-vector product (graph-based)
ED = np.zeros(shape=(DE.shape[0], DE.shape[0]))
for i,j in K['edges']:
  ED[i,j] = ED[j,i] = max(fv[i],fv[j])

x = np.random.uniform(size=DE.shape[0])[:,np.newaxis] 
y = np.zeros(DE.shape[0])
for i, xi in enumerate(x):
  J = np.array(list(G.neighbors(i)))
  y[i] = xi * (ED[i,J]**2).sum() - sum(x[J].flatten() * ED[i,J]**2)
  
## Check equivalent   
LE = DE @ DE.T
assert max(abs(y - (LE @ x).flatten())) < 1e-14

## O(m) mat-vec example for edge-weighted 1-Laplacians
z = np.zeros(DE.shape[0])
for cc, (i,j) in enumerate(K['edges']):
  e_ij = ED[i,j]**2
  z[i] += x[i]*e_ij - x[j]*e_ij
  z[j] += x[j]*e_ij - x[i]*e_ij
assert max(abs(z - ((DE @ DE.T) @ x).flatten())) < 1e-14

## Replace elementary chains w/ vertex-values
from scipy.sparse import csc_matrix
DV = boundary_matrix(K, p = 1).tocsc().astype(float)
RI, CI = DV.nonzero()
DV = DV.A
for ri, ci in zip(RI, CI):
  DV[ri,ci] = np.sign(DV[ri,ci]) * fv[ri]
DV = csc_matrix(DV)

## Show vertex-valued matrices
plt.imshow(abs(DV.A),interpolation='none',cmap='binary')
plt.imshow(abs((DV @ DV.T).A), interpolation='none',cmap='binary')

## Check that it matches the explicit form of the matrix-vector product (graph-based)
y = np.zeros(DV.shape[0])
for i, xi in enumerate(x.flatten()):
  J = np.array(list(G.neighbors(i)))
  y[i] = xi*G.degree(i)*fv[i]**2 - sum(x[J].flatten()*fv[i]*fv[J])
  
assert max(abs(((DV @ DV.T) @ x).flatten() - y)) < 1e-14

## Verify O(m) matrix-vec multiplication example 
D = np.array([G.degree(i) for i in G.nodes()])
z = np.zeros(DV.shape[0])
for i in range(D.shape[0]):
  z[i] += x[i]*D[i]*fv[i]**2
for cc, (i,j) in enumerate(K['edges']):
  f_ij = fv[i]*fv[j]
  z[i] -= x[j]*f_ij
  z[j] -= x[i]*f_ij

LV = DV @ DV.T
assert max(abs((LV @ x).flatten() - z)) < 1e-14

## Hadamard product in O(m) time
# DEV = np.sign(DE.A) * abs(DE * DV).A
# LEV = DEV @ DEV.T
# plt.imshow(abs(LEV), interpolation='none',cmap='binary')

## Form the correctly signed hadamard Laplacian
LEV = (DE * DV) @ (DE * DV).T
SM = np.sign(-np.ones(shape=LEV.shape) + np.diag(np.repeat(2, DE.shape[0])))
LEV = csc_matrix(SM * abs(LEV.todense()))

## Assert the signs are all correct
for i,j in zip(*LEV.nonzero()):
  assert LEV[i,j] < 0 if i != j else LEV[i,j] >= 0

## Assert the formula works for non-diagonal entries
for i,j in zip(*LEV.nonzero()):
  if i != j:
    assert (abs(LEV[i,j]) - (ED[i,j]**2 * fv[i] * fv[j])) < 1e-14

## O(m) solution 
z, Df = np.zeros(D.shape[0]), np.zeros(D.shape[0])
for cc, (i,j) in enumerate(K['edges']):
  Df[i] += ED[i,j]**2
  Df[j] += ED[i,j]**2
for i in range(D.shape[0]):
  z[i] += x[i]*Df[i]*fv[i]**2
# assert max(abs(np.diagonal(LEV @ np.diag(x.flatten())) - z)) < 1e-14
for cc, (i,j) in enumerate(K['edges']):
  z[i] -= x[j] * ED[i,j]**2 * fv[i] * fv[j]
  z[j] -= x[i] * ED[i,j]**2 * fv[i] * fv[j]

## YES! It works! 
assert max(abs((LEV @ x).flatten() - z)) < 1e-14

## Form the operator
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, aslinearoperator
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


## Verify the operator works 
LEV_matvec = aslinearoperator(UpLaplacianVE1_ls(K, fv))
assert all(max(abs(LEV_matvec.matvec(x) - (LEV @ x))) < 1e-12)

## Verify eiegnvalues are same 
from scipy.sparse.linalg import eigsh
LEV_lo = aslinearoperator(LEV)
ev1 = eigsh(LEV_matvec, k=5, return_eigenvectors=False)
ev2 = eigsh(LEV_lo, k=5, return_eigenvectors=False)
assert max(abs(ev1 - ev2)) < 1e-12

## Verify custom operator works
from pbsig.linalg import lanczos
I = np.array(K['edges'])[:,0].astype(int)
J = np.array(K['edges'])[:,1].astype(int)
v0 = np.random.uniform(size=len(fv), low=-0.50, high=0.50)
v0 = v0 / np.linalg.norm(v0)
res = lanczos.UL0_VELS_lanczos(fv, I, J, 5, 6, 1000, 1e-12, v0, 0, 0, 0, 0) # nev, num lanczos vectors, max_iter, tol
assert res['status'] == "success"
assert max(abs(np.sort(res['eigenvalues']) - np.sort(ev1))) < np.sqrt(np.finfo(np.float32).eps)

## Now adjust to match smoothstep 
from pbsig.betti import smoothstep
a,b,eps = np.quantile(fv, 0.25), np.quantile(fe, 0.75), np.finfo(float).eps
ss_b = smoothstep(lb = b-0.20, ub = b+eps, down = True)     
ss_ac = smoothstep(lb = a-0.20, ub = a+eps, down = False) 

DE = boundary_matrix(K, p = 1).tocsc()
DE.sort_indices()
DE.data = np.sign(DE.data)*ss_b(np.repeat(fe, 2))

DV = boundary_matrix(K, p = 1).tocsc().astype(float)
RI, CI = DV.nonzero()
DV = DV.A
for ri, ci in zip(RI, CI):
  DV[ri,ci] = np.sign(DV[ri,ci]) * ss_ac(fv[ri])
DV = csc_matrix(DV)

## Form the correctly signed hadamard Laplacian
LEV = (DE * DV) @ (DE * DV).T
SM = np.sign(-np.ones(shape=LEV.shape) + np.diag(np.repeat(2, DE.shape[0])))
LEV = csc_matrix(SM * abs(LEV.todense()))

class UpLaplacianVE1_ls(LinearOperator):
  def __init__(self, K: dict, fv: ArrayLike, a: float, b: float, w: float):
    eps = np.finfo(float).eps
    self.ss_b = smoothstep(lb = b-w, ub = b+eps, down = True)   # includes S(b) > 0, marks S(b+eps) = 0    
    self.ss_ac = smoothstep(lb = a-w, ub = a+eps, down = False) # set S(a-w)=0, S(a-w+eps)>0, and S(a+eps) = 1
    self.nv = len(fv)
    self.fv = fv
    self.fe = lambda i,j: max(self.fv[i],self.fv[j])
    self.z: ArrayLike = np.zeros(self.nv)
    self.Df: ArrayLike= np.zeros(self.nv)
    for cc, (i,j) in enumerate(K['edges']):
      self.Df[i] += self.ss_b(self.fe(i,j))**2
      self.Df[j] += self.ss_b(self.fe(i,j))**2
    self.shape = (int(self.nv), int(self.nv))
    self.dtype = np.dtype(np.float64)

  def _matvec(self, x: ArrayLike):
    self.z.fill(0.0)
    for i in range(self.nv):
      self.z[i] += x[i]*self.Df[i]*self.ss_ac(self.fv[i])**2
    for cc, (i,j) in enumerate(K['edges']):
      self.z[i] -= x[j] * self.ss_b(self.fe(i,j))**2 * self.ss_ac(self.fv[i]) * self.ss_ac(self.fv[j])
      self.z[j] -= x[i] * self.ss_b(self.fe(i,j))**2 * self.ss_ac(self.fv[i]) * self.ss_ac(self.fv[j])
    return self.z 

## Ensure operator work
LEV_matvec = aslinearoperator(UpLaplacianVE1_ls(K, fv, a=a, b=b, w=0.20))
assert all(max(abs(LEV_matvec.matvec(x) - LEV @ x)) < 1e-12)

## Ensure the eigenvalues are the same
ev1 = eigsh(LEV_matvec, k=8, return_eigenvectors=False)
ev2 = eigsh(LEV.todense(), k=8, return_eigenvectors=False)
assert max(abs(ev1 - ev2)) < 1e-12

## Try custom
# from pbsig.linalg import lanczos
# I = np.array(K['edges'])[:,0].astype(int)
# J = np.array(K['edges'])[:,1].astype(int)
# v0 = np.random.uniform(size=len(fv), low=-0.50, high=0.50)
# v0 = v0 / np.linalg.norm(v0)
# res = lanczos.UL0_VELS_lanczos(fv, I, J, 8, 9, 1000, 1e-12, v0, a, b, eps, 0.20) # nev, num lanczos vectors, max_iter, tol
# assert res['status'] == "success"
# assert max(abs(np.sort(res['eigenvalues']) - np.sort(ev1))) < np.sqrt(np.finfo(np.float32).eps)













from scipy.sparse import random
from scipy.sparse.linalg import eigsh, aslinearoperator
np.set_printoptions(precision=2, suppress=True)

M = random(20,20, density=0.15, random_state=1234)
A = aslinearoperator(M @ M.T)

print(eigsh(A, k=19, return_eigenvectors=False))
# 0.   0.   0.01 0.03 0.06 0.07 0.12 0.2  0.3  0.52 0.62 0.67 0.89 1.13 1.3 1.59 2.47 2.97 4.63

print(eigsh(A, sigma=4.0, k=2, return_eigenvectors=False, which='LM'))
# [2.97 4.63]
print(eigsh(A, sigma=2.40, k=2, return_eigenvectors=False, which='LM'))
# [2.47 2.97]
print(eigsh(A, sigma=1.50, k=2, return_eigenvectors=False, which='LM'))
# [1.3  1.59]
print(eigsh(A, sigma=1.20, k=2, return_eigenvectors=False, which='LM'))
# [1.13 1.3 ]

eigsh(LEV_lo, sigma=1.94037204/2, v0=eve[:,0], k=2, return_eigenvectors=True, which='LM')
## Augment to use smoothsteps


import petsc
dir(petsc)
import slepc4py
slepc4py.init()
format(dir(slepc4py.lib.ImportPETSc()))
format(dir(slepc4py.lib.ImportSLEPc()))
format(dir(slepc4py.SLEPc))

format(dir(slepc4py.SLEPc.EPS()))


z - LEV.diagonal()

for cc, (i,j) in enumerate(K['edges']):
  f_ij, e_ij = fv[i]*fv[j], ED[i,j]**2
  z[i] -= x[j]*e_ij*fv[i]*fv[j]
  z[j] -= x[i]*e_ij*fv[i]*fv[j]

np.diag(LEV)
np.diag(LEV @ np.eye(DE.shape[0]))

# max_spectral = np.sqrt(max([sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()]))

# max_laplacian_spectral = np.sqrt(2)*np.sqrt(max([G.degree(v)**2 + sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()])) 
#max(np.linalg.eigh(L)[0])

## Verify O(n) matrix-vec multiplication example 
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


lanczos.UL1_LS_lanczos(fv, E[:,0], E[:,1], 8, 9)



np.array(sorted(np.linalg.eigh((D @ D.T).A)[0][1:], reverse=True))

from scipy.sparse.linalg import eigsh
v0 = np.random.uniform(size=D.shape[0])[:,np.newaxis]
eigsh(D @ D.T, k=3, return_eigenvectors=False, maxiter=1, ncv=4, v0=v0,tol=np.inf)



## Generic testing of spectra's lanczos 
import numpy as np
X = np.random.uniform(size=(15,15))
X = X.T @ X

from pbsig.linalg import lanczos
from scipy.sparse import csc_matrix

A = csc_matrix(X)
v0 = np.random.uniform(A.shape[1])

wut = lanczos.sparse_lanczos(A, 4, 5, 19, 1e-10)


ev = np.array(sorted(np.linalg.eigh(A.A)[0], reverse=True))
shift = ev[1]
ev_shifted = np.array(sorted(np.linalg.eigh(A.A - shift*np.eye(A.shape[0]))[0], reverse=True))
ev_shifted+shift


from scipy.sparse.linalg import eigsh
# eigsh(A, k=A.shape[0]-1,return_eigenvectors=False, which="LM")

ev = eigsh(A, k=3, ncv=5, return_eigenvectors=False, which="LM")
shift = max(ev)
B = csc_matrix(A - shift*np.eye(A.shape[0]))
eigsh(B, k=3, return_eigenvectors=False, which="LM")+shift

#
# for i in range(1, 100):



lanczos.sparse_lanczos(A, 3, 4, 1, 0.10)['eigenvalues'] - lanczos.sparse_lanczos(A, 3, 4, 1000, 1e-12)['eigenvalues']

# wut = lanczos.sparse_lanczos(A, 4, 5, 19, 1e-10)


















# lanczos.sparse_lanczos(A, 1, 2, 10, 1e-10)
lanczos.UL1_LS_lanczos()



import matplot.pyplot as plt
from numpy.polynomial.legendre import Legendre
x,y = Legendre.fromroots([0.0, 0.5, 1.0, np.sqrt(2)], domain=[-1, 2]).linspace(1000)

plt.plot(x,y)


from pbsig.simplicial import delaunay_complex
from pbsig.persistence import boundary_matrix

X = np.random.uniform(size=(10,2))
K = delaunay_complex(X)
D1 = boundary_matrix(K, p = 1).tocsc()

import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(X.shape[0]))
G.add_edges_from(K['edges'])

A = nx.adjacency_matrix(G)
D = np.diag([G.degree(i) for i in range(X.shape[0])])

all(np.array((D - A) == (D1 @ D1.T).A).flatten())

## Check Lx; indeed it works! 
L = D - A
x = np.random.uniform(size=L.shape[0])
v0 = np.array([D[i,i]*x[i] - sum(x[A[i,:].indices]) for i, xi in enumerate(x)]).flatten()
v1 = (L @ x).flatten()

f = X @ np.array([0, 1])
E = K['edges']
fe = f[E].max(axis=1)
#a, b = np.median(f), np.median(fe)
a, b = f[3], fe[-3]
w = 0.50

from pbsig.betti import smoothstep
D1 = boundary_matrix(K, 1).tocsc()
D1.sort_indices()
eps = 1e-12
ss_b = smoothstep(lb = b-w, ub = b+eps, reverse = True)     # 1 (b-w) -> 0 (b), includes (-infty, b]
ss_ac = smoothstep(lb = a-w, ub = a+eps, reverse = False)   # 0 (a-w) -> 1 (a), includes (a, infty)
nz_pn = np.sign(D1.data)

entries = np.repeat(ss_b(fe), 2)*ss_ac(f[E].flatten())*nz_pn
entries[entries == 0.0] = 0.0
D1.data = entries

## finally: the laplacian
L = D1 @ D1.T
y = L @ x

np.set_printoptions(edgeitems=30, linewidth=100000)
np.around(L.A, 2)

## TODO: fix this
sum([f[i]**2 * ss_b(ew)**2 for (i,j), ew in zip(E, fe) if (i == 0 or j == 0)])


v0 = np.array([D[i,i]*x[i] - sum(x[A[i,:].indices]) for i, xi in enumerate(x)]).flatten()

# for v in range(D.shape[0]):
plot_mesh2D(X, K['edges'], K['triangles'])

# Vertex-weighted multiplication 
D1 = boundary_matrix(K, 1).tocsc()
D1.sort_indices()

## Functions on the vertices and edges
rf = np.linspace(0, D1.shape[0])*0.7581
cf = np.linspace(0, D1.shape[1])*0.6257

row_ind, col_ind = D1.nonzero()
D1_r = D1.copy()
D1_r[np.ix_(row_ind, col_ind)] = rf[row_ind]
L_r = D1_r @ D1_r.T

D1_c = D1.copy()
D1_c[np.ix_(row_ind, col_ind)] = cf[col_ind]
L_c = D1_c @ D1_c.T

L_r @ x
L_c @ x

deg = np.diagonal((D1 @ D1.T).A)
L = (D1 @ D1.T).A

## Lx (vertex case)
y = np.zeros(len(x))
for i, xi in enumerate(x):
  t1 = xi*deg[i]*rf[i]**2  
  J = np.setdiff1d(np.flatnonzero(L[i,:]), i)
  t2 = xi*sum(x[J]*rf[J])
  y[i] = t1 - t2 

X = np.random.uniform(size=(24,2))
K = delaunay_complex(X)

import networkx as nx 
G = nx.Graph()
G.add_nodes_from(K['vertices'])
G.add_edges_from(K['edges'])
vf = np.random.uniform(size=len(G.nodes))
ef = np.random.uniform(size=len(G.edges))

A = boundary_matrix(K, p=1).tocsc()
A.sort_indices()
ri, ci = A.indices, np.repeat(range(A.shape[1]), 2)

Ar, Ac = A.copy(), A.copy()
Ar.data, Ac.data = np.sign(A.data)*vf[ri], np.sign(A.data)*ef[ci]
Lr, Lc = (Ar @ Ar.T).A, (Ac @ Ac.T).A

x = np.random.uniform(size=Lr.shape[1])

## Lr @ x 
y = np.zeros(len(x))
for i, xi in enumerate(x):
  t1 = xi*G.degree(i)*vf[i]**2  
  J = np.array(list(G.neighbors(i)))
  t2 = vf[i]*sum(x[J]*vf[J])
  y[i] = t1 - t2

max(abs((Lr @ x) - y)) < 1e-12 # true 

## Lc @ x 
E = list(G.edges)
y = np.zeros(len(x))
for i, xi in enumerate(x):
  J = np.array(list(G.neighbors(i)))
  j_ind = np.array([E.index((i,j)) if i < j else E.index((j,i)) for j in J], dtype=int)
  y[i] = xi*sum(ef[j_ind]**2) - sum(x[J]*(ef[j_ind]**2))

max(abs((Lc @ x) - y)) < 1e-12 # true 

## Equivalent formulation(s) (faster)

## Lc @ x; assumes w(i,j) = w(j,i) 
y = np.zeros(len(x))
for i,j in G.edges():
  w_ij = ef[E.index((i,j)) if i < j else E.index((j,i))]**2
  y[i] += (x[i]*w_ij - x[j]*w_ij)
  y[j] += (x[j]*w_ij - x[i]*w_ij)

## Lr @ x; assumes w(i,j) = w(j,i) 
y = np.zeros(len(x))
for i, fi in enumerate(vf):
  y[i] = x[i]*G.degree(i)*fi**2
for i,j in G.edges():
  y[i] -= x[j]*vf[i]*vf[j]
  y[j] -= x[i]*vf[i]*vf[j]

max(abs((Lr @ x) - y)) 


## Test both
y = np.zeros(len(x))
for i, fi in enumerate(vf):
  y[i] = x[i]*G.degree(i)*fi**2
for i,j in G.edges():
  w_ij = ef[E.index((i,j)) if i < j else E.index((j,i))]**2
  y[i] += (x[i]*w_ij - x[j]*w_ij) - x[j]*vf[i]*vf[j]
  y[j] += (x[j]*w_ij - x[i]*w_ij) - x[i]*vf[i]*vf[j]

((Lc * Lr) @ x) - y



DN = np.diag(1/np.sqrt(np.diagonal(D)))

## Normalized laplacian 
LN = np.eye(X.shape[0]) - DN @ A @ DN
max(np.linalg.eigh(LN)[0]) # should be in [0,2]

## Spectrum + characteristic polynomial: http://web.stanford.edu/class/msande319/MatchingSpring19/lecture07.pdf
sA = np.linalg.eigh(A.A)[0] ## should all be [-maxdeg(G), maxdeg(G)]
cp = lambda x: np.prod(x - sA)
supp = np.linspace(np.min(sA), np.max(sA), 1000)


plt.plot(supp, [cp(x) for x in supp])
plt.scatter(sA, np.repeat(0, len(sA)), c='red')
plt.gca().set_ylim(-5, 5)


## Regular graph laplacian 
max(np.linalg.eigh((D1 @ D1.T).A)[0]) # not in [0,2]


L = (D1 @ D1.T).A
x = np.random.uniform(size=L.shape[0])[:,np.newaxis]


x.T @ L @ x 

sum([(x[i] - x[j])**2 for i,j in G.edges()])







