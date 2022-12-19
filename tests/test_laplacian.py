## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import delaunay_complex, graph_laplacian, edge_iterator
from pbsig.utility import lexsort_rows

## Generate random geometric complex
X = np.random.uniform(size=(15,2))
K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
D1, D2 = boundary_matrix(K, p=(1,2))

## Test unweighted/combinatorial quadratic form
x = np.random.uniform(low=-0.50, high=0.50, size=D1.shape[0])
v0 = sum([(x[i] - x[j])**2 for i,j in K.faces(1)])
v1 = x @ (D1 @ D1.T) @ x
assert np.isclose(v0, v1), "Unweighted Quadratic form failed!"

## Test unweighted/combinatorial linear form
ns = K.dim()
w0,w1 = np.ones(ns[0]), np.ones(ns[1])
r, v0 = np.zeros(ns[1]), np.zeros(ns[0])
for cc, (i,j) in enumerate(K.faces(1)):
  r[cc] = w1[cc]*w0[i]*x[i] - w1[cc]*w0[j]*x[j]
for cc, (i,j) in enumerate(K.faces(1)):
  v0[i] += w1[cc]*r[cc] #?? 
  v0[j] -= w1[cc]*r[cc]
v1 = (D1 @ D1.T) @ x
assert np.allclose(v0, v1), "Unweighted Linear form failed!"

## Test weighted linear form (only takes one O(n) vector!)
ns = K.dim()
w0,w1 = np.random.uniform(size=ns[0]), np.random.uniform(size=ns[1])
v0 = np.zeros(ns[0]) # note the O(n) memory
for cc, (i,j) in enumerate(K.faces(1)):
  v0[i] += w1[cc]**2
  v0[j] += w1[cc]**2
v0 = w0**2 * v0 * x # O(n)
for cc, (i,j) in enumerate(K.faces(1)):
  v0[i] -= x[j]*w0[i]*w0[j]*w1[cc]**2
  v0[j] -= x[i]*w0[i]*w0[j]*w1[cc]**2
v1 = (diags(w0) @ D1 @ diags(w1**2) @ D1.T @ diags(w0)) @ x
assert np.allclose(v0, v1), "Unweighted Linear form failed!"

# def up_laplacian(K: Union[SparseMatrix, SimplicialComplex], p: int = 0, normed=False, return_diag=False, form='array', dtype=None, **kwargs):
## Up Laplacian - test weighted linear form (only takes one O(n) vector!) 
ns = K.dim()
w0,w1 = np.random.uniform(size=ns[1]), np.random.uniform(size=ns[2])

P = np.sign((D2 @ D2.T).tocsc().data)
LW = (diags(w0) @ D2 @ diags(w1**2) @ D2.T @ diags(w0)).tocsc()
assert all(np.sign(L2.data) == P)

p_faces = list(K.faces(p=1))
z = np.zeros(ns[1]) 
for t_ind, t in enumerate(K.faces(p=2)):
  for e in t.boundary():
    e_ind = p_faces.index(e)
    z[e_ind] += w1[t_ind]**2 * w0[e_ind]**2
assert max(abs(LW.diagonal() - z)) < 1e-13

x = np.random.uniform(size=ns[1])
y = np.zeros(ns[1])
del_sgn_pattern = [1,-1,1]
lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(del_sgn_pattern,2)])
for t_ind, t in enumerate(K.faces(p=2)):
  for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
    ii, jj = p_faces.index(e1), p_faces.index(e2)
    c = w0[ii] * w0[jj] * w1[t_ind]**2
    y[ii] += x[jj] * c * s_ij
    y[jj] += x[ii] * c * s_ij

#assert max(abs((abs(L2) @ x) - (y + z*x))) <= 1e-13, "Failed signless comparison"
assert max(abs((L2 @ x) - (y + x*z))) <= 1e-13, "Failed matrix vector"

## Using only one O(n) vector!
x = np.random.uniform(size=ns[1])
y = np.zeros(ns[1])
del_sgn_pattern = [1,-1,1]
lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(del_sgn_pattern,2)])
for t_ind, t in enumerate(K.faces(p=2)):
  for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
    ii, jj = p_faces.index(e1), p_faces.index(e2)
    c = w0[ii] * w0[jj] * w1[t_ind]**2
    y[ii] += x[jj] * c * s_ij
    y[jj] += x[ii] * c * s_ij
for t_ind, t in enumerate(K.faces(p=2)):
  for e in t.boundary():
    ii = p_faces.index(e)
    y[ii] += x[ii] * w1[t_ind]**2 * w0[ii]**2
assert max(abs((LW @ x) - y)) <= 1e-13, "Failed matrix vector"

## Test the up-laplacian matrices and matrix free methods yield the same results 
from pbsig.linalg import up_laplacian
L1 = up_laplacian(K, w0=w0, w1=w1**2, p=1)
assert max(abs((L1 @ x) - y)) <= 1e-13

L1 = up_laplacian(K, w0=w0, w1=w1**2, p=1, form='lo')
assert max(abs((L1 @ x) - y)) <= 1e-13


## Apply a directional trasnform 
from pbsig.linalg import eigsh_family
from pbsig.pht import rotate_S1, uniform_S1

V, E = np.array(list(K.faces(0))).flatten(), np.array(list(K.faces(1)))
fv, fe = lambda v: (X @ v)[V] + 1.0, lambda v: (X @ v)[E].max(axis=1) + 1.0
L0_dt = (up_laplacian(K, fv(v), fe(v), p=0) for v in uniform_S1(32))
R = list(eigsh_family(L0_dt, p = 0.50))


## Make a directional transform class?

## Weights should be non-negative to ensure positive semi definite?
#np.linalg.eigvalsh(up_laplacian(K, p=0).todense())
#np.linalg.eigvalsh((diags(w0) @ up_laplacian(K, p=0) @ up_laplacian(K, p=0)).todense())


list(eigsh_family(L0_dt, p = 1.0, reduce=sum))

LO = next(L0_dt)
trace_threshold(LO, 1.0)

LO = up_laplacian(K, w0, w1, p=0, form='lo')

# I = np.zeros(LO.shape[0])
# d = np.zeros(LO.shape[0])
# for i in range(LO.shape[0]):
#   I[i] = 1
#   I[i-1] = 0
#   d[i] = (LO @ I)[i]

# I = I.tocsc()



[(I[:,[i]].T @ LO @ I[:,[i]]).data[0] for i in range(LO.shape[0])]

I[:,[i]].todense()

G = {
  'vertices' : np.ravel(np.array(list(K.faces(0)))),
  'edges' : np.array(list(K.faces(1))),
  'triangles' : np.array(list(K.faces(2)))
}

y + z*x

# np.sign(L2.tocsc().data)

v1 = L2 @ x


# e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv)) ## Can be cast using minimal perfect hashing function
# deg_e *= (fe**2)
assert np.allclose(deg_e - LT_ET.diagonal(), 0.0), "Collecting diagonal terms failed!"

y = np.zeros(len(x))
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv))
  for (ii,jj),s_ij in zip(combinations(e_ind, 2), [-1,1,-1]):
    ## The sign pattern is deterministic! 
    v = (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)
    y[ii] += x[jj] * v * s_ij
    y[jj] += x[ii] * v * s_ij






## Test linear form
L = graph_laplacian(K)
y = L.diagonal() * x # np.zeros(shape=len(x))
for i,j in K['edges']:
  y[i] -= x[j]
  y[j] -= x[i]
assert all(np.isclose(L @ x, y)), "Linear form failed!"

## Test edge-weighted version: LE = \hat{\partial_1}\hat{\partial_1}^T = \partial_1 W^2 \partial_1^T
fe = np.random.uniform(size=K['edges'].shape[0], low=0.0, high=1.0)
W = diags(fe)
D1_E = D1.copy().tocsc()
D1_E.data = np.sign(D1_E.data) * np.repeat(fe, 2)
LE = D1 @ W**2 @ D1.T
assert np.allclose((LE - (D1_E @ D1_E.T)).data, 0), "edge-weighed Laplacian form wrong!"

## Test edge-weighted matrix-vector product
d = LE.diagonal() # the row sums of LE not including diagonal
y = d * x
for cc, (i,j) in enumerate(K['edges']):
  y[i] -= fe[cc]**2 * x[j]
  y[j] -= fe[cc]**2 * x[i]
assert np.allclose(y, LE @ x), "edge-weighted Laplacian linear form wrong!"

## Test L_2^up entry-wise
nv = len(G['vertices'])
E, T = G['edges'], G['triangles']
S = (D2 @ D2.T).copy() ## Precompute sign matrix
S.data = np.sign(S.data)
fe = np.random.uniform(size=E.shape[0], low=0.0, high=1.0)
ft = np.random.uniform(size=T.shape[0], low=0.0, high=1.0)
D2_ET = D2.copy().tocsc()
D2_ET.data = np.sign(D2_ET.data) * ft[np.repeat(range(D2.shape[1]), 3)]*fe[D2_ET.indices]
LT_ET = D2_ET @ D2_ET.T 

## (a) Test entry-wise from weighted boundary matrix is (+/-1)*fi*fj*f(cofacet(i,j))**2
from pbsig.utility import rank_combs, rank_comb
cofacet = lambda i,j: np.sort(np.unique(np.append(E[i,:], E[j,:])))
tr = rank_combs(T, n=len(K['vertices']), k=3)
for i,j in edge_iterator(LT_ET):
  sigma_ind = np.searchsorted(tr, rank_comb(cofacet(i,j), n=nv, k=3)) # this could be done in O(1) time via minimal perfect hashing
  assert abs(LT_ET[i,j]-S[i,j]*fe[i]*fe[j]*ft[sigma_ind]**2) <= np.sqrt(1e-15)

## (b) Test the diagonal entries match fii**2 * (sum(f(cofacets(eii))**2))
cofacets = lambda i,j: T[np.logical_and(T[:,0] == i, T[:,1] == j) | np.logical_and(T[:,1] == i, T[:,2] == j) | np.logical_and(T[:,0] == i, T[:,2] == j),:] 
for cc, (i,j) in enumerate(E):
  sigma_ind = np.searchsorted(tr, rank_combs(cofacets(i,j), n=nv, k=3))
  #print(sum([(ft[s]*fe[cc])**2 for s in sigma_ind]))
  assert abs(LT_ET[cc,cc] - (fe[cc]**2)*sum(ft[sigma_ind]**2)) <= np.sqrt(1e-15)
  
## (c) Now test the matrix form LT_ET == We * D2 * Wt^2 * D2 * We
Wt, We = diags(ft), diags(fe)
assert np.allclose((LT_ET - (We @ D2 @ Wt**2 @ D2.T @ We)).data, 0)

## (d) Formulate the naive matrix-vector product that enumerates edges 

## (e) Formulate the O(m) matrix-vector product that enumerates triangles
from pbsig.utility import *
nv = len(G['vertices'])
E, T = G['edges'], G['triangles']
fe,ft = w0,w1
er = rank_combs(E, n=len(G['vertices']), k=2)
x = np.random.uniform(size=E.shape[0], low=0.0, high=1.0)
deg_e = np.zeros(len(x)) 
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv)) ## Can be cast using minimal perfect hashing function
  deg_e[e_ind] += ft[t_ind]**2 * (fe[e_ind]**2) ## added fe

# deg_e *= (fe**2)
assert np.allclose(deg_e - LT_ET.diagonal(), 0.0), "Collecting diagonal terms failed!"

y = np.zeros(len(x))
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv))
  for (ii,jj),s_ij in zip(combinations(e_ind, 2), [-1,1,-1]):
    ## The sign pattern is deterministic! 
    v = (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)
    y[ii] += x[jj] * v * s_ij
    y[jj] += x[ii] * v * s_ij
  # for ii in e_ind:
  #   y[ii] += x[jj] * (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)

    #y[jj] += x[ii] * (fe[jj]**2) * (ft[t_ind]**2)
  # for ii,jj in combinations(e_ind, 2):
  #   y[ii] += x[ii] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2
  #   y[jj] += x[jj] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2

assert max(abs((LT_ET @ x) - (y + deg_e*x))) <= np.sqrt(1e-16), "Weighted up-Laplacian matrix-vector product failed"
# (abs(LT_ET.todense()) @ x)[:5]

z = LT_ET @ x

s = rank_combs(K['edges'], n=nv, k=2) % 2

for i,j in combinations(range(nv), 2):
  if L2[i,j] != 0:
    print(f"sign: {np.sign(L2[i,j])}, i: {s[i]}, j: {s[j]}")

from pbsig.utility import parity
def boundary_sgn(face, coface):
  return parity(np.append(face, np.setdiff1d(coface, face)))



nv = len(K['vertices'])
E_ind = np.sort(rank_combs(K['edges'], k=2, n=nv))
z = np.zeros(len(x))
for t in K['triangles']:
  i,j,k = np.sort(t)
  ii,ji,ki = np.searchsorted(E_ind, rank_combs([[i,j],[i,k],[j,k]],k=2,n=nv))
  z[ii]
