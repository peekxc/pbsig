from typing import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb
from itertools import combinations

from combin import rank_to_comb, comb_to_rank
from splex import * 

def _h0_apparent_pairs(K: ComplexLike, f: Callable, refinement: str = "lex"):
  n = card(K, 0)
  if card(K,1) == 0: return []
  edges = np.array(list(faces(K,1))).astype(np.uint16)
  E_ranks = comb_to_rank(edges, n=n, order='lex')

  ## Store the initial list of apparent pair candidates
  pair_candidates = []

  ## Since n >> k in almost all settings, start by getting apparent pair candidates from the p+1 simplices
  for e in E_ranks:
    i, j = rank_to_comb(e, k=2, n=n, order='lex')
    facets = [[i], [j]]
    facet_weights = f(facets)
    same_value = np.isclose(facet_weights, f([i,j]))
    if any(same_value):
      ## Choose the "youngest" facet, which is the *maximal* in lexicographical order
      lex_min_ind = int(np.flatnonzero(same_value)[-1])
      pair_candidates.append((facets[lex_min_ind], [i,j]))
    
  ## Now filter the existing pairs via scanning through each p-face's cofacets
  true_pairs = []
  for v,e in pair_candidates:
    facet_weight = f(v)
    
    ## Find the "oldest" cofacet, which is the *minimal* in lexicographical order
    max_cofacet = None
    for k in range(n): # reversed for maximal
      ## NOTE: equality is necessary here! Using <= w/ small rips filtration yields 16 pairs, whereas equality yields 48 pairs. 
      if Simplex(k) != Simplex(v) and np.isclose(facet_weight, f(Simplex([v,k]))): 
        max_cofacet = Simplex((k,v))
        break
    
    ## If the relation is symmetric, then the two form an apparent pair
    if max_cofacet is not None and max_cofacet == Simplex(e):
      true_pairs.append((tuple([v]), max_cofacet))
  
  return true_pairs

def _h1_apparent_pairs(K: ComplexLike, f: Callable, refinement: str = "lex"):
  n = card(K, 0)
  if card(K,2) == 0: return []
  triangles = np.array(list(faces(K,2))).astype(np.uint16)
  T_ranks = comb_to_rank(triangles, n=n, order='lex')

  ## Store the initial list of apparent pair candidates
  pair_candidates = []

  ## Since n >> k in almost all settings, start by getting apparent pair candidates from the p+1 simplices
  for t in T_ranks:
    i, j, k = rank_to_comb(t, k=3, n=n, order='lex')
    facets = [[i,j], [i,k], [j,k]]
    facet_weights = f(facets)
    same_value = np.isclose(facet_weights, f([i,j,k]))
    if any(same_value):
      ## Choose the "youngest" facet, which is the *maximal* in lexicographical order
      lex_min_ind = int(np.flatnonzero(same_value)[-1])
      pair_candidates.append((facets[lex_min_ind], [i,j,k]))
    
  ## Now filter the existing pairs via scanning through each p-face's cofacets
  true_pairs = []
  for e,t in pair_candidates:
    i,j = e
    facet_weight = f(e)
    
    ## Find the "oldest" cofacet, which is the *minimal* in lexicographical order
    max_cofacet = None
    for k in range(n): # reversed for maximal
      ## NOTE: equality is necessary here! Using <= w/ small rips filtration yields 16 pairs, whereas equality yields 48 pairs. 
      if k != i and k != j and np.isclose(facet_weight, f([i,j,k])): 
        max_cofacet = Simplex((i,j,k))
        break
    
    ## If the relation is symmetric, then the two form an apparent pair
    if max_cofacet is not None and max_cofacet == Simplex(t):
      true_pairs.append((tuple(e), max_cofacet))
  
  return true_pairs

def apparent_pairs(K: ComplexLike, f: Callable, p: int = 0, refinement: str = "lex"):
  """Finds the H1 apparent pairs of lexicographically-refined clique filtration.

  A persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the youngest facet of sigma 
    2. sigma is the oldest cofacet of tau 
    3. the pairing has persistence |f(sigma)-f(tau)| = 0 
  
  Parameters: 
    K: Simplicial complex.
    f: filter function defined on K.
    refinement: the choice of simplexwise refinement. Only 'lex' is supported for now. 

  Returns: 
    pairs (e,t) with zero-persistence in the H1 persistence diagram.

  Details: 
    Observe tau is the facet of sigma with the largest filtration value, i.e. f(tau) >= f(tau') for any tau' \in facets(sigma)
    and sigma is cofacet of tau with the smallest filtration value, i.e. f(sigma) <= f(sigma') for any sigma' \in cofacets(tau). 
    There are potentially several cofacets of a given tau, thus to ensure uniqueness, this function assumes the 
    filtration induced by f is a lexicographically-refined simplexwise filtrations. 
    
    Equivalently, for lexicographically-refined simplexwise filtrations, we have that a 
    zero-persistence pair (tau, sigma) is said to be *apparent* iff: 
      1. tau is the lexicographically *maximal* facet of sigma w/ f(tau) = f(sigma)
      2. sigma is the lexicographically *minimal* cofacet of sigma w/ f(sigma) = f(tau)

    Note that Bauer define similar notions but under the reverse colexicographical ordering, in which case the notions 
    of minimal and maximal are reversed.

    What is known about apparent pairs: 
      - Any apparent pair in a simplexwise filtration is a persistence pair, regardless of the choice of coefficients
      - Apparent pairs often form a large portion of the total number of persistence pairs in clique filtrations. 
      - Apparent pairs of a simplexwise filtrations form a discrete gradient in the sense of Discrete Morse theory 

    Moreover, if K is a Rips complex and all pairwise distances are distinct, it is knonw that the persistent pairs 
    w/ 0 persistence of K in dimension 1 are precisely the apparent pairs.
  """
  if p == 0: 
    return _h0_apparent_pairs(K,f,refinement)
  elif p == 1: 
    return _h1_apparent_pairs(K,f,refinement)
  else: 
    raise NotImplementedError("Haven't implemented higher AP calculations")

  # result = []
  # for T in K['triangles']:  
  #   T_facets = comb_to_rank(combinations(T, 2), k=2, n=len(K['vertices']), order="lex")
  #   max_facet = T_facets[np.max(np.flatnonzero(d[T_facets] == np.max(d[T_facets])))] # lexicographically maximal facet 
  #   n = len(K['vertices'])
  #   u, v = rank_to_comb(max_facet, k=2, n=n)
  #   same_diam = np.zeros(n, dtype=bool)
  #   for j in range(n):
  #     if j == u or j == v: 
  #       continue
  #     else: 
  #       cofacet = np.sort(np.array([u,v,j], dtype=int))
  #       cofacet_diam = np.max(np.array([d[comb_to_rank(face, k=2, n=n, order="lex")] for face in combinations(cofacet, 2)]))
  #       if cofacet_diam == d[max_facet]:
  #         same_diam[j] = True
  #   if np.any(same_diam):
  #     j = np.min(np.flatnonzero(same_diam))
  #     cofacet = np.sort(np.array([u,v,j], dtype=int))
  #     if np.all(cofacet == T):
  #       pair = (max_facet, comb_to_rank(cofacet, k=3, n=n, order="lex"))
  #       result.append(pair)
  # ap = np.array(result)
  # return(ap)

## From ripser paper: 
# pair_totals = np.array([18145, 32167, 230051, 2192209, 1386646, 122324, 893])
# non_ap = np.array([ 53, 576, 2466, 14006, 576, 438, 39 ])
# (pair_totals-non_ap)/pair_totals
# array([0.99707909, 0.98209345, 0.98928064, 0.99361101, 0.99958461, 0.99641935, 0.95632699])

