from typing import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb
from itertools import combinations

from combin import rank_to_comb, comb_to_rank

def apparent_pairs(d: ArrayLike, K: List):
  """
  Given a vector of pairwise distances 'd', a flag complex 'K', and a homological dimension 0 <= p < 2, 
  this function identifies the apparent pairs (\sigma_p, \sigma_{p+1}) 

  TODO: should be given some epsilon + distances, enumerate edges instead of triangles and find cofacets, not rips complex

  A persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the youngest facet of sigma 
    2. sigma is the oldest cofacet of tau 
    3. the pairing has persistence |f(sigma)-f(tau)| = 0 

  Observe tau is the facet of sigma with the largest filtration value, i.e. f(tau) >= f(tau') for any tau' \in facets(sigma)
  and sigma is cofacet of tau with the smallest filtration value, i.e. f(sigma) <= f(sigma') for any sigma' \in cofacets(tau). 
  There are potentially several cofacets of a given tau, to ensure uniqueness, we rely on lexicographically-refined 
  simplexwise filtrations.

  Equivalently, for lexicographically-refined simplexwise filtrations, we have that a 
  zero-persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the lexicographically maximal facet of sigma w/ f(tau) = f(sigma)
    2. sigma is the lexicographically minimal cofacet of sigma w/ f(sigma) = f(tau)

  What is known about apparent pairs: 
    - Any apparent pair in a simplexwise filtration is a persistence pair, regardless of the choice of coefficients
    - Apparent pairs of a simplexwise filtrations form a discrete gradient in the sense of Discrete Morse theory 
    - If K is a Rips complex and all pairwise distances are distinct, the persistent pairs w/ 0 persistence of K 
      in dimension 1 are precisely the apparent pairs
  
  Empirically, it also known that apparent pairs form a large portion of the total number of persistence pairs. 
  """
  result = []
  for T in K['triangles']:  
    T_facets = rank_combs(combinations(T, 2), k=2, n=len(K['vertices']))
    max_facet = T_facets[np.max(np.flatnonzero(d[T_facets] == np.max(d[T_facets])))] # lexicographically maximal facet 
    n = len(K['vertices'])
    u, v = unrank_comb(max_facet, k=2, n=n)
    same_diam = np.zeros(n, dtype=bool)
    for j in range(n):
      if j == u or j == v: 
        continue
      else: 
        cofacet = np.sort(np.array([u,v,j], dtype=int))
        cofacet_diam = np.max(np.array([d[rank_comb(face, k=2, n=n)] for face in combinations(cofacet, 2)]))
        if cofacet_diam == d[max_facet]:
          same_diam[j] = True
    if np.any(same_diam):
      j = np.min(np.flatnonzero(same_diam))
      cofacet = np.sort(np.array([u,v,j], dtype=int))
      if np.all(cofacet == T):
        pair = (max_facet, rank_comb(cofacet, k=3, n=n))
        result.append(pair)
  ap = np.array(result)
  return(ap)

# pair_totals = np.array([18145, 32167, 230051, 2192209, 1386646, 122324, 893])
# non_ap = np.array([ 53, 576, 2466, 14006, 576, 438, 39 ])
# (pair_totals-non_ap)/pair_totals
# array([0.99707909, 0.98209345, 0.98928064, 0.99361101, 0.99958461, 0.99641935, 0.95632699])

# ## Test unrank works
# C = np.array([unrank_comb(c, k=3, n=10) for c in range(int(comb(10,3)))])
# np.all(np.array(list(combinations(range(10), 3))) == C)

# ## Test rank works
# np.all(np.array([rank_comb(c, 3, 10) for c in C]) == np.array(range(C.shape[0])))


