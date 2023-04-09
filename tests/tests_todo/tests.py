import numpy as np
from scipy.spatial.distance import pdist
from persistence import *
import persistence as pers

X = np.random.uniform(size=(15, 2))
K = rips(X, r=np.max(pdist(X))*0.15, d=2)

D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'])
D2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'])

# D1 = D1.astype(bool)
# D2 = D2.astype(bool)

## Regular reduction algorithm
R1, R2, V1, V2 = reduction_pHcol(D1, D2)
print(pers._perf['n_col_adds'])
is_reduced(R1)
is_reduced(R2)
np.sum((D1 @ V1) - R1) == 0.0
np.sum((D2 @ V2) - R2) == 0.0
np.allclose(V1.A, np.triu(V1.A))
np.allclose(V2.A, np.triu(V2.A))

## pHcol w/ clearing
R, V = reduction_pHcol_clearing([D2, D1])
print(pers._perf['n_col_adds'])
is_reduced(R[0])
is_reduced(R[1])
np.sum((D1 @ V[1]) - R[1]) == 0.0
np.sum((D2 @ V[0]) - R[0]) == 0.0
np.allclose(V[0].A, np.triu(V[0].A))
np.allclose(V[1].A, np.triu(V[1].A))


## Coboundary matrix construction 
C1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'], coboundary=True)
C2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'], coboundary=True)
R, V = reduction_pHcol_clearing([C2, C1])
print(pers._perf['n_col_adds'])
is_reduced(R[0])
is_reduced(R[1])
np.sum((C1 @ V[1]) - R[1]) == 0.0
np.sum((C2 @ V[0]) - R[0]) == 0.0
np.allclose(V[1].A, np.triu(V[1].A))
np.allclose(V[0].A, np.triu(V[0].A))


from scipy.sparse import lil_matrix, isspmatrix
D1L = lil_matrix(D1)



## weighted boundary matrices
for i in range(25):
  X = np.random.uniform(size=(25, 2), high=10)
  K = rips(X, r=np.max(pdist(X))*0.35, d=2)

  D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'])
  D2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'])
  b = np.max(pdist(X))/10.0
  edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]


  D1.data = D1.data*(b - np.repeat(edge_weights,2))
  D1 = D1[:, np.argsort(edge_weights)]

  R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=False)

  assert np.max(np.linalg.svd(D1.A)[1]) >= np.max(np.linalg.svd(R1.A)[1])


D1 = D1.astype(float)
eg_val, eg_vec = eigs(D1.T @ D1)
eg_val, eg_vec = eigs(D1 @ D1.T)

# spectral norm contained in [0, d+1]
# for edges (-1, 1) \sigma_max(D1.T @ D1) = \sigma_max(D1 @ D1.T) \in [0, np.min(D1.shape)]

# bounds for spectral norm: 
# 2*np.max(D1) + np.max(np.diag((D1 @ D1.T).A))
[(D1[j,:] @ D1[j,:].T).data.item() for j in range(D1.shape[1])]

# For CSR 
np.max([np.sum(D1[j,:].data**2) for j in range(D1.shape[0])])





def rips_persistence(X: ArrayLike, r: float):
  K = rips(X, r=r, d=2)
  D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'], coboundary=True)
  D2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'], coboundary=True)
  edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]
  D1.data = D1.data*(b - np.repeat(edge_weights,2))
  D1 = D1[:, np.argsort(edge_weights)]
  R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)


# %% expanding circle 
import matplotlib.pyplot as plt
n = 18
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
circle = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(n,2), low=0.10, high=0.25)
b, d = 1.0, 1.5
plt.scatter(*circle.T)
plt.gca().set_aspect('equal')

# Given b, find upper bound on when t*S1 yields disconnected set of points
from scipy.optimize import root_scalar
lb = lambda t: np.max(pdist(circle @ np.diag([t, t]))) - b
ub = lambda t: np.min(pdist(circle @ np.diag([t, t]))) - b
max_t = root_scalar(ub, bracket=(0.01, 5.0)).root
min_t = root_scalar(lb, bracket=(0.01, 5.0)).root

ranks, nnorms = [], []
for t in np.linspace(min_t, max_t, 150):
  X = circle @ np.diag([t, t])
  K = rips(X, r=b/2.0, d=2)
  E, T = len(K['edges']), len(K['triangles'])
  D1 = weighted_H1(X, K, b)
  D2 = weighted_H2(X, K, d)
  # R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=False)
  D1_rank = 0 if np.prod(D1.shape) == 0 else np.linalg.matrix_rank(D1.A)
  D2_rank = 0 if np.prod(D2.shape) == 0 else np.linalg.matrix_rank(D2.A)
  D1_nn = np.sum(abs(np.linalg.svd(D1.A)[1]))
  ranks.append(D1_rank)
  nnorms.append(D1_nn)
  print(f"D1 rank: {D1_rank}, D2 rank: {D2_rank}")

## Bounds on spectral norm    
# M = np.sqrt((n*(n-1)/2)* b**2) # frobenius norm bound
# M = (2+n-1)*b # naive using degree
# M = 2*np.max(abs(b - edge_weights)) + np.max(np.diag(D1.A @ D1.A.T)) # bound using largest known weighted degree
# M = np.max(np.linalg.svd(D1.A)[1]) # As-is: assuming D1 has largest spectral norm

## Convex envelope plot 
plt.plot(ranks)
M = np.sqrt((n*(n-1)/2)* b**2) # frobenius norm bound
plt.plot(np.array(nnorms)/M, label="Frobenius bound")
M = (2+n-1)*b 
plt.plot(np.array(nnorms)/M, label="Laplacian degree bound")

X = circle @ np.diag([min_t, min_t])
K = rips(X, r=b/2.0, d=2)
D1 = weighted_H1(X, K, b)
M = 2*np.max(np.abs(D1.data)) + np.max(np.diag(D1.A @ D1.A.T)) 
plt.plot(np.array(nnorms)/M, label="Laplacian degree bound (weighted)")
M = np.max(np.linalg.svd(D1.A)[1]) 
plt.plot(np.array(nnorms)/M, label="Tightest possible")
plt.gca().legend()




## Gradient optimization
def subgradient(X: ArrayLike):
  U, S, Vt = np.linalg.svd(X, full_matrices = False)
  return(U @ Vt)

def weighted_H1(X: ArrayLike, K: List, b: float, sorted=False):
  """ K := Rips complex """
  E, T = len(K['edges']), len(K['triangles'])
  D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'])
  edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]
  D1.data = D1.data*np.maximum(b - np.repeat(edge_weights,2), 0)
  if sorted: 
    ind = np.argsort(edge_weights)
    D1 = D1[:,ind]
    edge_weights = edge_weights[ind]
  return(D1, edge_weights)

def weighted_H2(X: ArrayLike, K: List, b: float, sorted=False):
  n, D = X.shape[0], pdist(X)
  tri_weight = lambda T: np.max([D[rank_C2(T[0],T[1],n)], D[rank_C2(T[0],T[2],n)], D[rank_C2(T[1],T[2],n)]])
  tri_weights = np.array([tri_weight(tri) for tri in K['triangles']])
  D2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'], N = X.shape[0])
  D2.data = D2.data*np.maximum(b - np.repeat(tri_weights,3), 0)
  if sorted: 
    ind = np.argsort(tri_weights)
    D2 = D2[:,ind]
    tri_weights = tri_weights[ind]
  return(D2, tri_weights)

def nullspace_orthbasis(X: ArrayLike):
  U, S, Vt = np.linalg.svd(X, full_matrices = True)
  r = np.sum(abs(S) > 10*np.finfo(float).eps)
  return((Vt.T)[:,r:])

def colspace_orthbasis(X: ArrayLike):
  U, S, Vt = np.linalg.svd(X, full_matrices = False)
  r = np.sum(abs(S) > 10*np.finfo(float).eps)
  return(U[:,:r])

X = f(min_t)
b, d = 1.0, 1.5
K = rips(X, r=d/2.0, d=2)
D1, ew = weighted_H1(X, K, b)
D2, tw = weighted_H2(X, K, d)

Z_b = nullspace_orthbasis(D1.A)
B_d = colspace_orthbasis(D2.A)
P_zb  = Z_b @ Z_b.T 
P_bd  = B_d @ B_d.T 
M_bd = np.linalg.matrix_power(P_zb @ P_bd, 50)


from autograd import jacobian
def f(t): return(circle @ np.eye(2)*t)

b, d = 1.0, 1.5

ranks, nnorms = [], []
for i, t in enumerate(np.linspace(min_t, max_t, 50)):
  X = f(t)
  K = rips(X, r=d/2.0, d=2)
  D1, ew = weighted_H1(X, K, b, sorted=True)
  D2, tw = weighted_H2(X, K, d, sorted=True)
  R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)
  max_low = np.max(np.flatnonzero(ew <= b))
  bd_ind = np.flatnonzero(np.bitwise_and(low_entry(R2) != -1, low_entry(R2) <= max_low))
  ranks.append(len(bd_ind))
  nnorms.append(np.sum(abs(np.linalg.svd(R2[:,bd_ind].A)[1])))
  print(i)

plt.plot(ranks)
M = np.sqrt((n*(n-1)/2)* b**2) # frobenius norm bound
plt.plot(np.array(nnorms)/M, label="Frobenius bound")
M = (2+n-1)*b 
plt.plot(np.array(nnorms)/M, label="Laplacian degree bound")

X = f(min_t)
K = rips(X, r=d/2.0, d=2)
D2, tw = weighted_H2(X, K, d)
M = 2*np.max(np.abs(D2.data)) + np.max(np.diag(D2.A @ D2.A.T)) 
plt.plot(np.array(nnorms)/M, label="Laplacian degree (weighted)")
plt.gca().legend()

def M_bd(t):
  X = f(t)
  K = rips(X, r=d/2.0, d=2)
  D1, ew = weighted_H1(X, K, b, sorted=True)
  D2, tw = weighted_H2(X, K, d, sorted=True)
  R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)
  max_low = np.max(np.flatnonzero(ew <= b))
  bd_ind = np.flatnonzero(np.bitwise_and(low_entry(R2) != -1, low_entry(R2) <= max_low))
  return(R2[:,bd_ind].A)

# from autograd import jacobian
# J_bd = jacobian(M_bd)
# J_bd(min_t)

m1 = M_bd(min_t)

# ew = -(D1.max(axis=0).data-b) # edge weights, ordered
# tw = -(D2.max(axis=0).data-d)
# subgradient(D1.A).shape
# subgradient(M_bd).shape

import numpy as np
from ripser import Rips
rips_complex = Rips()
diagrams = rips_complex.fit_transform(X)
rips_complex.plot(diagrams)

(D1.shape, D2.shape)
# apparent pairs cannot be right
len(apparent_pairs(pdist(X), K))


K = rips(X, r=0.50, d=2)
D = pdist(X)
apparent_pairs(D, K)
(D1.shape, D2.shape)



ap = np.unique(np.array(apparent_pairs(pdist(X), K)), axis=0)
np.sum(pdist(X)[ap[:,0]] <= b)


# %%
import numpy as np
s = np.flip(np.sort(np.random.uniform(size=10)))
# w = np.sort(np.random.uniform(size=len(s), low=1.0, high=2.0))
W = lambda alpha: (1-alpha) + alpha*(1/s)

s*W(0.0)

# %%
