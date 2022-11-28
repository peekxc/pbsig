


import numpy as np 
from itertools import combinations

## Tests whether we know how to apply weighted matrix-vector product in closed form 

from pbsig.persistence import boundary_matrix
from pbsig.simplicial import delaunay_complex

X = np.random.uniform(size=(15,2))
K = delaunay_complex(X)

D1, D2 = boundary_matrix(K, p=(1,2))

x = np.random.uniform(low=-0.50, high=0.50, size=D1.shape[0])
x @ (D1 @ D1.T) @ x

x = np.random.uniform(low=-0.50, high=0.50, size=D2.shape[0])
L2 = D2 @ D2.T
y = (D2 @ D2.T) @ x

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
