## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import delaunay_complex
from pbsig.utility import lexsort_rows

## Generate random geometric complex
X = np.random.uniform(size=(15,2))
K = delaunay_complex(X)
D1, D2 = boundary_matrix(K, p=(1,2))

## Test quadratic form
x = np.random.uniform(low=-0.50, high=0.50, size=D1.shape[0])
v0 = sum([(x[i] - x[j])**2 for i,j in K['edges']])
v1 = x @ (D1 @ D1.T) @ x
assert np.isclose(v0, v1), "Quadratic form failed!"

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
nv = len(K['vertices'])
E, T = K['edges'], K['triangles']
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
er = rank_combs(E, n=len(K['vertices']), k=2)
x = np.random.uniform(size=E.shape[0], low=0.0, high=1.0)
y = np.zeros(len(x))
deg_e = np.zeros(len(x)) 
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv)) ## Can be cast using minimal perfect hashing function
  deg_e[e_ind] += ft[t_ind]**2
deg_e *= (fe**2)
assert np.allclose(deg_e - LT_ET.diagonal(), 0.0), "Collecting diagonal terms failed!"

for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv))
  y[e_ind] += x[e_ind] * (fe[e_ind]**2) * (ft[t_ind]**2)
  # for ii,jj in combinations(e_ind, 2):
  #   y[ii] += x[ii] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2
  #   y[jj] += x[jj] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2

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
