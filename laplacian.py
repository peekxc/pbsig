import numpy as np
np.set_printoptions(suppress=True)
#   b
#  /|\ 
# a | d
#  \|/ 
#   c

D1 = np.array([
  [0.1, 0.2, 0.0, 0.0, 0.0],
  [-0.1, 0.0, 0.3, 0.4, 0.0],
  [0.0, -0.2, -0.3, 0.0, 0.5],
  [0.0, 0.0, 0.0, -0.4, -0.5]
])

D1 @ D1.T

D2 = np.array([
  [0.3, 0.0],
  [-0.3, 0.0],
  [0.3, 0.5],
  [0.0, -0.5], 
  [0.0, 0.5]
])


D1_b = (D1.astype(bool).astype(int)*np.sign(D1)).astype(int)

D1_b @ D1_b.T

A = np.sign(D1 @ D1.T).astype(bool).astype(int)
np.fill_diagonal(A, 0)
D = np.diag([2,3,3,2])
D - A

(D - A) == (D1_b @ D1_b.T)

D2_b = (D2.astype(bool).astype(int)*np.sign(D2)).astype(int)
np.linalg.eigh(D2_b.T @ D2_b)[0]

LG = np.array([[1], [-1]])
LG @ LG.T

np.linalg.eigh(LG @ LG.T)[0]

LG = np.array([[0.5], [-0.5]])
np.linalg.eigh(LG @ LG.T)[0]



D2 = np.array([
  [1, 0, 0],
  [0, 0, 0],
  [-1, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [1, 0, -1],
  [0, 0, 1],
  [0, 1, 0],
])

from scipy.sparse import csc_matrix
K = {
  'vertices': np.array([1,2,3,4,5,6])-1, 
  'edges': np.array([[1,2],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5],[4,6],[5,6]])-1,
  'triangles' : np.array([[1,2,5],[2,3,4],[2,3,5],[2,4,5],[3,4,5],[4,5,6]])-1
}
D1 = boundary_matrix(K, p=1).A
D2 = boundary_matrix(K, p=2).A

L1 = D2 @ D2.T + D1.T @ D1 # r(A + B) <= r(A) + r(B)
np.sort(np.linalg.eigh(L1)[0])

np.sort(np.linalg.eigh(D1.T @ D1)[0])

np.sort(np.linalg.eigh(D2.T @ D2)[0])
np.sort(np.linalg.eigh(D2 @ D2.T)[0])

np.linalg.matrix_rank(D2 @ D2.T)
np.linalg.matrix_rank(D2.T @ D2)
np.linalg.matrix_rank(L1)

np.linalg.matrix_rank(D1 @ D1.T)


## 
import networkx as nx

nx.random

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator

# A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal'
A = np.array([
  [0.1, 0.2, 0.0, 0.0, 0.0],
  [-0.1, 0.0, 0.3, 0.4, 0.0],
  [0.0, -0.2, -0.3, 0.0, 0.5],
  [0.0, 0.0, 0.0, -0.4, -0.5]
])
D = csc_matrix(A @ A.T)
w,v = eigsh(D, k=1)

eigsh(D, k=1, v0=v, which='LM', ncv=2, maxiter=1, return_eigenvectors=False)


import numpy as np 
import networkx as nx
from pbsig import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

G = nx.fast_gnp_random_graph(n=10, p=0.30)
X = np.random.uniform(size=(10,2))

## lot X 
nx.draw(G, pos=X)

ER = enclosing_radius(squareform(pdist(X)))
K = rips(X, p=2, diam=ER)
D1 = boundary_matrix(K, p=1)
D2 = boundary_matrix(K, p=2)

PD = pdist(X)
e_weights = rips_weights(PD, faces=K['edges'])
t_weights = rips_weights(PD, faces=K['triangles'])

theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
V = np.c_[np.cos(theta), np.sin(theta)] # directions
Filtration_values = [X @ v[:,np.newaxis] for v in V]
  
## Plot showing filtration values over time
fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.gca()
for i, f in enumerate(Filtration_values):
  ax.scatter(np.repeat(i, len(f)), f, s=1.50)
ax.set_aspect('equal')

## Build the persistence diagrams
from pbsig.betti import *
dgms = [lower_star_ph_dionysus(scale_diameter(X), X @ v[:,np.newaxis],  K['triangles']) for v in V]



# from scipy.spatial import Delaunay
# from scipy.spatial.distance import pdist, cdist, squareform

# del_tri = Delaunay(X)
# V, T = scale_diameter(X), del_tri.simplices
# project_v = lambda X, v: (X @ np.array(v)[:,np.newaxis]).flatten()
# project_v(X, V[0,:])




