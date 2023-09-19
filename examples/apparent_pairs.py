import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from splex import *
from more_itertools import first_true
from pbsig.persistence import ph, generate_dgm
from pbsig.apparent_pairs import apparent_pairs

## Random geometric rips complex 
X = np.random.uniform(size=(14,2))
f = flag_weight(pdist(X))
er = enclosing_radius(pdist(X))
K = rips_complex(X, radius=er, p=2)

## Filter by fla weight and do full ph validation 
F = filtration(K, f)
R, V = ph(F, output="rv", validate=True)
D = boundary_matrix(F)
assert all(((D @ V) == R).data) 

## Check apparent pairs
dgm = generate_dgm(F, R, collapse=False, simplex_pairs=True)
ap = apparent_pairs(K, f)
all_pairs_dgm = sorted([(tuple(cr),tuple(de)) for (birth,death,cr,de) in dgm[1]])
ap_dgm = sorted([(tuple(cr),tuple(de)) for (birth,death,cr,de) in dgm[1] if np.isclose(death-birth, 0.0)])
ap_com = sorted([(tuple(c),tuple(d)) for c,d, in ap])
assert ap_com == ap_dgm

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



