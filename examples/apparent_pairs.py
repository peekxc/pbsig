import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from splex import *
from more_itertools import first_true

X = np.random.uniform(size=(20,2))
f = flag_weight(pdist(X))
er = enclosing_radius(pdist(X))
K = rips_complex(X, radius=er, p=2)

## TODO: check which one is correct
from pbsig.apparent_pairs import apparent_pairs
from pbsig.persistence import ph, generate_dgm
S = dict(vertices=list(faces(K,0)), edges=list(faces(K,1)), triangles=list(faces(K,2)))
len(apparent_pairs(pdist(X), S))


F = filtration(K, f)
R, V = ph(F, output="rv")
f([0,1,7])

## (0,1), (0,1,7) reported as pair in peristence diagram 
## but apparent pairs reports ((0,1), (0,1,8))
## They all have some function value and thus zero persistence
## latter is lexicographically maximal; check ripser paper again 

## Let's jump implement the example from Ripser paper

# %% 
dgm = generate_dgm(F, R, collapse=False, simplex_pairs=True)
dgm[1]['creators'] = [tuple(s) for s in dgm[1]['creators']]
dgm[1]['destroyers'] = [tuple(s) for s in dgm[1]['destroyers']]

np.sort(dgm[1], order=['creators'])


from splex import SimplexTree
simplices = [[0], [1], [2], [3], [2,3], [0,1], [1,3], [0,2], [0,3], [1,2], [1,2,3]]
simplices += [[0,2,3], [0,1,3], [0,1,2], [0,1,2,3]]
simplices = [tuple(s) for s in simplices]
f_values = [0]*4 + [3]*2 + [4]*2 + [5]*7
f = lambda s: f_values[simplices.index(tuple(s))]
K = filtration(simplices, f=f)

R, V = ph(K, output="rv")

generate_dgm(K, R, collapse=False, simplex_pairs=True)[1]

# F.reindex(np.arange(len(F)))
# dgm_index = generate_dgm(F, R, collapse=False, simplex_pairs=True)
# dgm_index[1]


ap = apparent_pairs_H1(K, np.vectorize(f))


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


## --- H1 apparent pairs ---
def apparent_pairs_H1(K: ComplexLike, f: Callable):
  """Finds the H1 apparent pairs of lexicographically-refined clique filtration.

  Parameters: 
    K: Simplicial complex.
    f: filter function defined on K.
  
  Returns: 
    pairs (e,t) with zero-persistence in the H1 persistence diagram.
  """
  import combin 
  from combin import comb_to_rank, rank_to_comb
  n = card(K, 0)
  triangles = np.array(list(faces(K,2))).astype(np.uint16)
  T_ranks = comb_to_rank(triangles, n=n, order='lex')

  ## Store the initial list of apparent pair candidates
  pair_candidates = []

  ## Since n >> k in almost all settings, start by getting apparent pair candidates from the p+1 simplices
  for t in T_ranks:
    i, j, k = rank_to_comb(t, k=3, n=20, order='lex')
    facets = [[i,j], [i,k], [j,k]]
    facet_weights = f(facets)
    same_value = np.isclose(facet_weights, f([i,j,k]))
    if any(same_value):
      ## Choose the lexicographically minimal facet 
      lex_min_ind = int(np.flatnonzero(same_value)[0])
      pair_candidates.append((facets[lex_min_ind], [i,j,k]))
    
  ## Now filter the existing pairs via scanning through each p-face's cofacets
  true_pairs = []
  for e,t in pair_candidates:
    i,j = e
    facet_weight = f(e)
    
    ## Find the lexicographically maximal cofacet
    max_cofacet = None
    for k in reversed(range(n)):
      if k != i and k != j and np.isclose(f([i,j,k]), facet_weight):
        max_cofacet = (i,j,k)
        break
    
    ## If the relation is symmetric, then the two form an apparent pair
    if max_cofacet is not None and tuple(max_cofacet) == tuple(t):
      true_pairs.append((tuple(e), max_cofacet))
  
  return true_pairs


# E_cofacets = []
# for e in E_ranks:
#   i, j = rank_to_comb(e, k=2, n=20, order='lex')
#   cofacet_ranks = np.array([comb_to_rank((i,j,k), n=n, order='lex') for k in range(n) if k != i and k != j], dtype=np.uint64) # lex order 
#   cofacets_revlex = rank_to_comb(np.flip(cofacet_ranks), k=3, n=n, order='lex')
#   candidates = np.argwhere(f(cofacets_revlex) == f((i,j)))
#   maximal_cofacet = cofacet_ranks[-candidates[0][0]] if len(candidates) > 0 else None
#   E_cofacets.append(maximal_cofacet)



