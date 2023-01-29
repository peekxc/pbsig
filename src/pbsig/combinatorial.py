import numpy as np 
from typing import * 
from itertools import * 
from math import comb, factorial
import _combinatorial as comb_mod

# comb_mod.rank_combs(np.array([0,1,2], dtype=int), 3, 10)

# bisect.bisect_left(range(n), idx, lo=0, hi=n, key=lambda x: comb(x,k))

def rank_C2(i: int, j: int, n: int) -> int:
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int) -> tuple:
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

# def max_vertex(idx: int, k: int, n: int) -> int:
#    ## locates insertion point for x in a
#   return get_max(n, k - 1, comb(w, k) <= idx)
#   # get_max_vertex(idx, k, n)

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
  index = np.sum([comb(cc, kk) for cc,kk in zip((n-1)-c, np.flip(range(1, k+1)))])
  return(int((comb(n, k)-1) - int(index)))

def rank_combs(C: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([rank_C2(c[0], c[1], n) for c in C], dtype=int))
  else: 
    return(np.array([rank_comb(c, k, n) for c in C], dtype=int))


def inverse_choose(x: int, k: int):
	assert k >= 1, "k must be >= 1" 
	if k == 1: return(x)
	if k == 2:
		rng = np.array(list(range(int(np.floor(np.sqrt(2*x))), int(np.ceil(np.sqrt(2*x)+2) + 1))))
		final_n = rng[np.nonzero(np.array([comb(n, 2) for n in rng]) == x)[0].item()]
	else:
		# From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
		if x < 10**7:
			lb = (factorial(k)*x)**(1/k)
			potential_n = np.array(list(range(int(np.floor(lb)), int(np.ceil(lb+k)+1))))
			idx = np.nonzero(np.array([comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[idx]
		else:
			lb = np.floor((4**k)/(2*k + 1))
			C, n = factorial(k)*x, 1
			while n**k < C: n = n*2
			m = (np.nonzero( np.array(list(range(1, n+1)))**k >= C )[0])[0].item()
			potential_n = np.array(list(range(int(np.max([m, 2*k])), int(m+k+1))))
			ind = np.nonzero(np.array([comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[ind]
	return(final_n)

from bisect import bisect_left, bisect_right
from functools import cmp_to_key

def all_longest_subseq(x: Sequence[int]):
  assert all(np.sort(x) == np.arange(len(x)))

  pred = lambda y,i: max(y[y < i]) if any(y < i) else None
  succ = lambda y,i: min(y[y > i]) if any(y > i) else None
  E = np.array([], dtype=int)       # T data structure
  L = np.zeros(len(x), dtype=int)   # L[i] := length of LIS ending in i

  ## Stage 1: compute length of the LIS that ends on x[i]
  for i, m in enumerate(x):
    E = np.append(E, m)
    if pred(E, m) is not None: 
      L[m] = L[pred(E,m)]+1
    else:
      L[m] = 1
    if succ(E, m) is not None and L[succ(E,m)] == L[m]:
      E = E[E != succ(E, m)]
  Q = np.argsort(x) # inverse permutation 
  L = L[x] ## because our permutations are word permutations
  assert all(L == L[x][Q])

  ## Get left and right successor/predecessor info
  L = L.astype(int)
  L1 = [x[max(np.flatnonzero(L[:j] == l))] if any(L[:j] == l) else None for j,l in enumerate(L) ] # =
  L2 = [x[max(np.flatnonzero(L[:j] == l-1))] if any(L[:j] == l-1) else None for j,l in enumerate(L) ]  # -1
  
  ## Recursively enumerate LIS's
  k = max(L) # all LIS lengths
  results = set()
  def enum_lis(z: int, out: Sequence[int]):
    if L[Q[z]] == k or z < out[L[Q[z]]]: # (L[z]+1 <= k and
      out[L[Q[z]]-1] = z
    else: 
      return 
    z1 = L2[Q[z]]
    if z1 is None:
      results.add(tuple(out))
      # print(out)
      return # I think this is needed
    else:
      enum_lis(z1, out)
    while z1 is not None and L1[Q[z1]] is not None:
      enum_lis(L1[Q[z1]], out)
      z1 = L2[Q[z1]]
  out = np.array([x[max(np.flatnonzero(L == i))] for i in range(1, k+1)])
  z = out[-1]
  for z in reversed(x): enum_lis(z, out)
  return results

## From: https://stackoverflow.com/questions/3992697/longest-increasing-subsequence
def longest_subsequence(seq, mode='strictly', order='increasing', key=None, index=False):
  bisect = bisect_left if mode.startswith('strict') else bisect_right

  # compute keys for comparison just once
  rank = seq if key is None else map(key, seq)
  if order == 'decreasing':
    rank = map(cmp_to_key(lambda x,y: 1 if x<y else 0 if x==y else -1), rank)
  rank = list(rank)

  if not rank: return []

  lastoflength = [0] # end position of subsequence with given length
  predecessor = [None] # penultimate element of l.i.s. ending at given position

  for i in range(1, len(seq)):
    # seq[i] can extend a subsequence that ends with a lesser (or equal) element
    j = bisect([rank[k] for k in lastoflength], rank[i])
    # update existing subsequence of length j or extend the longest
    try: lastoflength[j] = i
    except: lastoflength.append(i)
    # remember element before seq[i] in the subsequence
    predecessor.append(lastoflength[j-1] if j > 0 else None)

  # trace indices [p^n(i), ..., p(p(i)), p(i), i], where n=len(lastoflength)-1
  def trace(i):
    if i is not None:
      yield from trace(predecessor[i])
      yield i
  indices = trace(lastoflength[-1])

  return list(indices) if index else [seq[i] for i in indices]