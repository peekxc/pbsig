import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from splex import faces, rips_complex, lower_star_weight, flag_weight, enclosing_radius
from more_itertools import first_true

X = np.random.uniform(size=(20,2))
f = flag_weight(pdist(X))
er = enclosing_radius(pdist(X))
K = rips_complex(X, radius=er, p=2)

## --- H1 apparent pairs ---
## Step 1: find the cofacets of each edge in reverse rexicographical order
import combin 
from combin import comb_to_rank, rank_to_comb
n = K.n_simplices[0]
E_ranks = comb_to_rank(K.edges.astype(np.uint16), n=K.n_simplices[0], order='lex')
T_ranks = comb_to_rank(K.triangles.astype(np.uint16), n=K.n_simplices[0], order='lex')


E_cofacets = []
for e in E_ranks:
  i, j = rank_to_comb(e, k=2, n=20, order='lex')
  cofacet_ranks = np.array([comb_to_rank((i,j,k), n=n, order='lex') for k in range(n) if k != i and k != j], dtype=np.uint64) # lex order 
  cofacets_revlex = rank_to_comb(np.flip(cofacet_ranks), k=3, n=n, order='lex')
  candidates = np.argwhere(f(cofacets_revlex) == f((i,j)))
  maximal_cofacet = cofacet_ranks[-candidates[0][0]] if len(candidates) > 0 else None
  E_cofacets.append(maximal_cofacet)

## Why wouldn't you always just start by getting apparent pair candidates with triangles
T_facets = []
for t in T_ranks:
  i, j, k = rank_to_comb(t, k=3, n=20, order='lex')

  facet_ranks = rank_to_comb(np.flip(cofacet_ranks), k=3, n=K.n_simplices[0], order='lex')

  f()

