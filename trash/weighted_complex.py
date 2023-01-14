# %%
import numpy as np
import matplotlib.pyplot as plt
from persistence import * 
from apparent_pairs import * 
theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
circle = np.c_[np.cos(theta), np.sin(theta)]
r = np.min(pdist(circle))*0.55
K = rips(circle, r, d=1)
fig, axs = plt.subplots(1, 2)
axs[0].scatter(*circle.T)
for u, v in K['edges']: axs[1].plot(*circle[(u,v),:].T)
axs[1].scatter(*circle.T)
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')

# %%
from apparent_pairs import *
from typing import *

def boundary_jacobian(K, f, t: float, h: float = 100*np.sqrt(np.finfo(float).eps)):
  D = boundary_matrix(K, p=1)
  edge_ind = rank_combs(K['edges'], n=len(K['vertices']), k=2)
  ew_p = np.maximum(2.0 - pdist(f(t + h))[edge_ind], 0.0)
  ew_m = np.maximum(2.0 - pdist(f(t - h))[edge_ind], 0.0)
  ew_p = np.sign(D.data) * np.repeat(ew_p, 2)
  ew_m = np.sign(D.data) * np.repeat(ew_m, 2)
  D.data = (ew_p - ew_m)/(2*h)
  return(D)

# %% Show 1-parameter family persistent Betti numbers vs relaxation
from persistence import weighted_H1, weighted_H2
f = circle_family(16)

X = np.random.uniform(size=(16, 2))
K = rips(X, d=2, diam=np.inf) # full complex (2-skeleton)
D1, ew = weighted_H1(X, K, sorted=True)
D2, tw = weighted_H2(X, K, sorted=True)

## Add time-varying p-chains
b, d = 1.0, 1.5


# ap = apparent_pairs(pdist(f(1.0)), K)

## Given b,d: collect the terms

# a. Count the number of simplices w/ diam(sigma) <= b
t1 = np.sum(pdist(X) <= b)

# b. Get the rank of 1st boundary matrix w/ weighted p-chains 
D1, ew = weighted_H1(X, K, sorted=True)
D1.data = np.sign(D1.data) * np.repeat(np.maximum(b - ew, 0.0), 2)
t2 = np.linalg.matrix_rank(D1.A)

# c. Get the intersection term 
D1, ew = weighted_H1(X, K, sorted=True)
D2, tw = weighted_H2(X, K, sorted=True)
D1.data = np.sign(D1.data) * np.repeat(np.maximum(d - ew, 0.0), 2)
D2.data = np.sign(D2.data) * np.repeat(np.maximum(d - tw, 0.0), 3)
R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)
max_low = np.max(np.flatnonzero(ew <= b))
bd_ind = np.flatnonzero(np.bitwise_and(low_entry(R2) != -1, low_entry(R2) < max_low))
t3 = len(bd_ind)

b_ind = np.max(np.flatnonzero(ew <= b))
d_ind = np.max(np.flatnonzero(tw <= d))
np.linalg.matrix_rank(D2[:,:d_ind].A)-np.linalg.matrix_rank(D2[(b_ind+1):,:d_ind].A)


# Test cases
K = {
  'vertices' : list(range(5)), 
  'edges' : [(0,1), (0,3), (0,4), (1,2), (1,4), (2,3), (2,4), (3,4)],
  'triangles' : [(0,3,4), (1,2,4)]
}

persistent_betti(D1, D2, i=8, j=0)
persistent_betti(D1, D2, i=7, j=0)


from persistence import *

f = circle_family(8)
X = f(1.0)
persistent_betti_rips(X, 1.65, 1.90) # issue = 1.90 check w/ smaller example
plot_rips(X, diam=1.90)



# # d1_sgn = np.sign(D1.data) # save non-zero pattern
# # D1.data = d1_sgn * np.repeat(np.maximum(b - ew, 0.0), 2)
# # t2 = np.linalg.matrix_rank(D1.A)

# # c. Get the intersection term 

# # D1.data = d1_sgn * np.repeat(np.maximum(d - ew, 0.0), 2)
# # D2.data = np.sign(D2.data) * np.repeat(np.maximum(d - tw, 0.0), 3)
# # b_ind = np.max(np.flatnonzero(ew <= b))
# # d2_r1 = 0 if np.prod(D2.shape) == 0 else np.linalg.matrix_rank(D2.A)
# # d2_r2 = 0 if np.prod(D2[(b_ind+1):,:].shape) == 0 else np.linalg.matrix_rank(D2[(b_ind+1):,:].A)
# # t3 = d2_r1-d2_r2

# t1 = np.sum(ew <= b)
# return(t1 - (t2 + t3))


D1 = boundary_matrix(K, 1)
D2 = boundary_matrix(K, 2)
persistent_betti(D1, D2, i=8, j=0)

b,d = 0.42, 0.43
K = rips(X, d = 2)
persistent_betti(X, K, 0.42, 0.43)

X = np.array([[0,0], [1,0], [0,2], [1,2], [1,0]], dtype=float)
K = rips(X, d = 2)
persistent_betti(X, K, 0.42, 0.43)

persistent_betti_rips(X, K, 1.0, 1.5)

plot_rips(X, 1.00, figsize=(2.5,2.5), dpi=180)

plot_rips(X, 0.40, figsize=(2.5,2.5), dpi=180, poly_opt={'alpha': 0.50})
plot_rips(X, 0.42, figsize=(2.5,2.5), dpi=180, poly_opt={'alpha': 0.50})
plot_rips(X, 0.43, figsize=(2.5,2.5), dpi=180, poly_opt={'alpha': 0.50})

np.sum(abs(weighted_H1(K, f, 0.01).data))

# boundary_jacobian(K, f, 2.0).A

(# %%
# np.linalg.svd() + Jacobian 


# %%
# D1, ew = weighted_H1(X, K, b, sorted=True)
# D2, tw = weighted_H2(X, K, d, sorted=True)
# R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)


