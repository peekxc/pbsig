import numpy as np
from pbsig.datasets import noisy_circle
from pbsig.vis import figure_complex
from simplextree import SimplexTree
from pbsig.shape import landmarks
from scipy.spatial.distance import pdist, cdist, squareform
from itertools import combinations

X = noisy_circle()
landmark_ind, ins_radii = landmarks(X, X.shape[0])
insert_map = np.zeros(len(X))
insert_map[landmark_ind] = ins_radii

dx = squareform(pdist(X))
eps = 0.90
def edge_birth_time(e: tuple):
  i, j = e
  min_insert = min(insert_map[i], insert_map[j])
  if (dx[i,j] <= (2*min_insert*(1+eps))/eps):
    return dx[i,j] / 2
  elif (dx[i,j] <= ((insert_map[i] + insert_map[j])*(1 + eps))/eps):
    return dx[i,j] - min_insert * (1+eps) / eps
  else: 
    return np.inf

# perturb_dist = np.array([edge_birth_time(e) for e in combinations(range(len(X)), 2)])

def simplex_birth_time(sigma: tuple):
  if len(sigma) == 1: 
    return 0.0
  elif len(sigma) == 2: 
    return edge_birth_time(sigma)
  elif len(sigma) == 3: 
    e1 = edge_birth_time(sigma[(0,1)])
    e2 = edge_birth_time(sigma[(0,2)])
    e3 = edge_birth_time(sigma[(1,2)])
    min_birth = np.min(insert_map[sigma] * (1 + eps)**2 / eps)
    max_weight = np.max([e1,e2,e3]) 
    return max_weight if max_weight <= min_birth else np.inf
  else: 
    return np.inf

from splex import filter_weight
f_sparse = filter_weight(simplex_birth_time)
X_weight = f_sparse(combinations(range(len(X)), 2))

np.sum(X_weight == np.inf)

## TODO: vary epsilon in 0 < eps < 1 for all edges and triangles, verify discontinuous behavior 


