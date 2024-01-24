import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from splex import *
from more_itertools import first_true
from pbsig.persistence import ph, generate_dgm
from pbsig.apparent_pairs import apparent_pairs

#%% Random geometric rips complex 
np.random.seed(1234)
X = np.random.uniform(size=(18,2))

f = flag_filter(pdist(X))
er = enclosing_radius(pdist(X))
K = rips_complex(X, radius=er, p=2)

# %% Filter by rips weight and do full ph validation 
dx = squareform(np.array([[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]]))
K = rips_complex(dx, p = 3)
f = flag_filter(dx)
F = rips_filtration(dx, p=3)
# F = filtration(K, f)
R, V = ph(F, output="rv", validate=True)
D = boundary_matrix(F)
assert all(((D @ V) == R).data) 

# %%  Check apparent pairs are indeed persistent pairs
dgm = generate_dgm(F, R, collapse=False, simplex_pairs=True)
ap = apparent_pairs(K, f, p = 1)
all_pairs_dgm = sorted([(tuple(cr),tuple(de)) for (birth,death,cr,de) in dgm[1]])
ap_dgm = sorted([(tuple(cr),tuple(de)) for (birth,death,cr,de) in dgm[1] if np.isclose(death-birth, 0.0)])
ap_com = sorted([(tuple(c),tuple(d)) for c,d, in ap])
assert ap_com == ap_dgm

# %% Show the diagram
from pbsig.vis import figure_dgm, show
show(figure_dgm(dgm[1]))

# %% Randomly choose lower-left matrices and compare their ranks
from itertools import product
N = R.shape[0]

def exact_rank(A):
  if np.prod(A.shape) == 0: 
    return 0
  elif len(A.data) == 0 or np.all(A.data == 0): 
    return 0
  else:
    return np.linalg.matrix_rank(A.todense())
  
from more_itertools import random_product
def check_lower_left(D, R, k: int):
  for ii in range(k):
    i, j = random_product(range(N), range(N))
    rank_Rij = exact_rank(R[i:,:j])
    rank_Dij = exact_rank(D[i:,:j])
    if rank_Rij != rank_Dij:
      print(f"{i}, {j}")
      assert rank_Rij == rank_Dij

check_lower_left(D, R, 50)

# %% See if removing apparent pairs has the intended effect on the rank
D_coo = D.tocoo()
R_coo = R.tocoo()
def set_rc_zero(A, i: int, mode: int):
  if mode == 0:
    A.data[A.row == i] = 0
    A.data[A.col == i] = 0
  elif mode == 1: 
    A.data[A.col == i] = 0
  elif mode == 2: 
    A.data[A.row == i] = 0
  return A

print(f"Rank D: {np.linalg.matrix_rank(D_coo.todense())}")
print(f"Rank R: {np.linalg.matrix_rank(R_coo.todense())}")

## Loop through apparent (p, p+1) pairs
## looks like removing the creator (p)-columns works, but the can't remove their rows
for ps, qs in ap:
  # ps, qs = next(iter(ap))
  drank, rrank = exact_rank(D_coo), exact_rank(R_coo)
  
  ## Ensure rank is unchanged / creators in nullspace  
  D_coo = set_rc_zero(D_coo, F.index(ps), mode=1)
  R_coo = set_rc_zero(R_coo, F.index(ps), mode=1)
  assert drank == exact_rank(D_coo) and rrank == exact_rank(R_coo)
  #assert np.all(D_coo.todense()[F.index(ps),:] == 0)
  #assert np.all(D_coo.todense()[:,F.index(ps)] == 0)

  ## Ensure pivots decrease the rank 
  # D_coo = set_rc_zero(D_coo, F.index(qs))
  # R_coo = set_rc_zero(R_coo, F.index(qs))
  
  # No longer true that rank D and rank R are the same
  ## Removing zero from R does decrease it rank but not so with D
  check_lower_left(D_coo.tolil(), R_coo.tolil(), 50)

  #check_lower_left(D_coo.tolil(), R_coo.tolil(), 500)

# 
D_coo.eliminate_zeros()



from pbsig.persistence import low_entry
pivots = low_entry(R_coo)
piv_ind = np.flatnonzero(pivots != -1)

for c, r in zip(piv_ind, pivots[piv_ind]):
  assert R_coo

import splex as sx
D1 = sx.boundary_matrix(F, p=1).todense() % 2 
D2 = sx.boundary_matrix(F, p=2).todense() % 2 

from pbsig.persistence import pHcol
from scipy.sparse import lil_array

R1 = lil_array(D1.astype(np.float64))
V1 = lil_array(np.eye(D1.shape[1]).astype(np.float64))
pHcol(R=R1, V=V1)
R1.todense()
V1.todense()


# %% Ripser example 
from scipy.spatial.distance import pdist, squareform
F = rips_filtration(squareform(np.array([[0,3,4,5],[3,0,5,4],[4,5,0,3],[5,4,3,0]])), p=3)
D1 = sx.boundary_matrix(F, p=1) 
D2 = sx.boundary_matrix(F, p=2).todense() % 2

np.linalg.matrix_rank(D2[-3:,:3]) 
np.linalg.matrix_rank(D2[-3:,:2]) 
np.linalg.matrix_rank(D2[-2:,:3]) 
np.linalg.matrix_rank(D2[-2:,:2])



R2 = lil_array(D2.astype(np.float64))
V2 = lil_array(np.eye(D2.shape[1]).astype(np.float64))
pHcol(R=R2, V=V2)
R2.todense() % 2

## This is not gaurenteed to be true 
# for p in range(2):
#   for c,d in zip(dgm[p]['creators'], dgm[p]['destroyers']):
#     assert (Simplex(c) <= Simplex(d)) or np.isnan(d[0])



## (0,1), (0,1,7) reported as pair in peristence diagram 
## but apparent pairs reports ((0,1), (0,1,8))
## They all have some function value and thus zero persistence
## latter is lexicographically maximal; check ripser paper again 

## Let's jump implement the example from Ripser paper


## TODO: check which one is correct
# from pbsig.apparent_pairs import apparent_pairs

# S = dict(vertices=list(faces(K,0)), edges=list(faces(K,1)), triangles=list(faces(K,2)))
# len(apparent_pairs(pdist(X), S))

# %% 
dgm = generate_dgm(F, R, collapse=False, simplex_pairs=True)
dgm[1]['creators'] = [tuple(s) for s in dgm[1]['creators']]
dgm[1]['destroyers'] = [tuple(s) for s in dgm[1]['destroyers']]

np.sort(dgm[1], order=['destroyers', 'creators'])

## dgm def wrong
ap_dgm = sorted(list(zip(dgm[1]['creators'], dgm[1]['destroyers'])))
ap_comp = sorted(ap)

np.sum(dgm[1]['birth'] == dgm[1]['death']) == len(ap)

from splex import is_complex_like
from splex import SimplexTree
simplices = [[0], [1], [2], [3], [2,3], [0,1], [1,3], [0,2], [0,3], [1,2], [1,2,3]]
simplices += [[0,2,3], [0,1,3], [0,1,2], [0,1,2,3]]
simplices = [tuple(s) for s in simplices]
f_values = [0]*4 + [3]*2 + [4]*2 + [5]*7
f = lambda s: np.array([f_values[simplices.index(Simplex(si))] for si in s]) if is_complex_like(s) else f_values[simplices.index(Simplex(s))]
K = filtration(simplices, f=f)

R, V = ph(K, output="rv")

generate_dgm(K, R, collapse=False, simplex_pairs=True)[1]

# F.reindex(np.arange(len(F)))
# dgm_index = generate_dgm(F, R, collapse=False, simplex_pairs=True)
# dgm_index[1]


ap = apparent_pairs_H1(K, f)


# apparent_pairs(pdist(X), K)
# result = []
# for T in K['triangles']:  
#   T_facets = rank_combs(combinations(T, 2), k=2, n=len(K['vertices']))
#   max_facet = T_facets[np.max(np.flatnonzero(d[T_facets] == np.max(d[T_facets])))] # lexicographically maximal facet 
#   n = len(K['vertices'])
#   u, v = unrank_comb(max_facet, k=2, n=n)
#   same_diam = np.zeros(n, dtype=bool)
#   for j in range(n):
#     if j == u or j == v: 
#       continue
#     else: 
#       cofacet = np.sort(np.array([u,v,j], dtype=int))
#       cofacet_diam = np.max(np.array([d[rank_comb(face, k=2, n=n)] for face in combinations(cofacet, 2)]))
#       if cofacet_diam == d[max_facet]:
#         same_diam[j] = True
#   if np.any(same_diam):
#     j = np.min(np.flatnonzero(same_diam))
#     cofacet = np.sort(np.array([u,v,j], dtype=int))
#     if np.all(cofacet == T):
#       pair = (max_facet, rank_comb(cofacet, k=3, n=n))
#       result.append(pair)
# ap = np.array(result)
# return(ap)


# E_cofacets = []
# for e in E_ranks:
#   i, j = rank_to_comb(e, k=2, n=20, order='lex')
#   cofacet_ranks = np.array([comb_to_rank((i,j,k), n=n, order='lex') for k in range(n) if k != i and k != j], dtype=np.uint64) # lex order 
#   cofacets_revlex = rank_to_comb(np.flip(cofacet_ranks), k=3, n=n, order='lex')
#   candidates = np.argwhere(f(cofacets_revlex) == f((i,j)))
#   maximal_cofacet = cofacet_ranks[-candidates[0][0]] if len(candidates) > 0 else None
#   E_cofacets.append(maximal_cofacet)



