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

import networkx as nx
G = nx.Graph()
G.add_nodes_from(K['vertices'])
G.add_edges_from(K['edges'])
max_spectral = np.sqrt(max([sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()]))

max_laplacian_spectral = np.sqrt(2)*np.sqrt(max([G.degree(v)**2 + sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()])) 
#max(np.linalg.eigh(L)[0])

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







