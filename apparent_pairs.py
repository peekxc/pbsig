from typing import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb
from itertools import combinations
from persistence import *

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 



def unrank_comb(r: int, k: int, n: int):
  result = np.zeros(k, dtype=int)
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return(result)

def unrank_combs(R: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([unrank_C2(r, n) for r in R], dtype=int))
  else: 
    return(np.array([unrank_comb(r, k, n) for r in R], dtype=int))

def rank_comb(c: Tuple, k: int, n: int):
  c = np.array(c, dtype=int)
  index = np.sum(comb((n-1)-c, np.flip(range(1, k+1))))
  return(int((comb(n, k)-1) - index))

def rank_combs(C: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([rank_C2(c[0], c[1], n) for c in C], dtype=int))
  else: 
    return(np.array([rank_comb(c, k, n) for c in C], dtype=int))



def apparent_pairs(D: ArrayLike, K: List):
  """
  Given a set of pairwise distances 'D', a Rips complex 'K', and a homological dimension 0 <= p < 2, 
  this function identifies the apparent pairs (\sigma_p, \sigma_{p+1}) 

  TODO: should be given some epsilon + distances, enumerate edges and find cofacets, not rips complex

  An zero-persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the youngest facet of sigma 
    2. sigma is the oldest cofacet of tau 
  
  Observe tau is the facet of sigma with the largest filtration value, i.e. f(tau) >= f(tau') for any tau' \in facets(sigma)
  and sigma is cofacet of tau with the smallest filtration value, i.e. f(sigma) <= f(sigma') for any sigma' \in cofacets(tau). 
  There are potentially several cofacets of a given tau, to ensure uniqueness, we rely on lexicographically-refined 
  simplexwise filtrations.

  Equivalently, for lexicographically-refined simplexwise filtrations, we have that a 
  zero-persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the lexicographically maximal facet of sigma w/ diam(tau) = diam(sigma)
    2. sigma is the lexicographically minimal cofacet of sigma w/ diam(sigma) = diam(tau)

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
    max_facet = T_facets[np.max(np.flatnonzero(D[T_facets] == np.max(D[T_facets])))] # lexicographically maximal facet 
    n = len(K['vertices'])
    u, v = unrank_comb(max_facet, k=2, n=n)
    same_diam = np.zeros(n, dtype=bool)
    for j in range(n):
      if j == u or j == v: 
        continue
      else: 
        cofacet = np.sort(np.array([u,v,j], dtype=int))
        cofacet_diam = np.max(np.array([D[rank_comb(face, k=2, n=n)] for face in combinations(cofacet, 2)]))
        if cofacet_diam == D[max_facet]:
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


