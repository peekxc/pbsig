from __future__ import division
from multiprocessing.sharedctypes import Value
from typing import * 
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_lowercase
from itertools import combinations
from pbsig.vineyards import *

index_of = lambda X, x: np.flatnonzero(X == x).item()
pairwise = lambda C: zip(C, C[1:])

## Choose two random permutations + y-coords to begin with
n = 12
p = np.random.choice(range(n), size=n, replace=False) # in order of f0! 
q = np.random.choice(range(n), size=n, replace=False) # in order of f1!
f0 = np.sort(np.random.uniform(size=n, low=0, high=1))
f1 = np.sort(np.random.uniform(size=n, low=0, high=1))

## Make a map to make it clear
F0 = { c : f for c, f in zip(p, f0) }
F1 = { c : f for c, f in zip(q, f1) }

## Plot lines
fig = plt.figure(figsize=(8,4), dpi=320)
ax = fig.gca()
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.scatter(np.repeat(0, n), f0, s=3.5)
ax.scatter(np.repeat(1, n), f1, s=3.5)
all((ax.text(-0.025, y, s=c, ha='center', va='center') for c, y in F0.items()))
all((ax.text(1.025, y, s=c, ha='center', va='center') for c, y in F1.items()))
for i in range(n):
  pp = (0, F0[i])
  qq = (1, F1[i])
  ax.plot(*np.vstack((pp, qq)).T, c='blue', linewidth=0.5)

from pbsig.vineyards import inversion_dist
sum(is_discordant(combinations(range(n), 2), p, q))
sum([pair_is_discordant(sa,sb,list(p),list(q)) for sa, sb in combinations(range(n), 2)])
inversion_dist(p,q)

## The algorithm
tr = []
tr_s = []
while any(p != q):
  cross_f = np.zeros(n-1) ## how early does a pair cross
  tau_dist = inversion_dist(p, q)
  for i, (sa, sb) in enumerate(pairwise(p)):
    pair_dc = pair_is_discordant(sa,sb,list(p),list(q))
    if pair_dc and inversion_dist(swap(p, i),q) < tau_dist: # discordant pair
      int_pt = intersection(line((0, F0[sa]), (1, F1[sa])), line((0, F0[sb]), (1, F1[sb])))
      assert int_pt[0] >= 0.0 and int_pt[0] <= 1.0
      cross_f[i] = np.inf if (int_pt[0] < 0 or int_pt[0] > 1) else int_pt[0]
    else: 
      cross_f[i] = np.inf
  assert not(all(cross_f == np.inf))
  ci = np.argmin(cross_f)
  if ci != np.inf:
    tr.append(ci)
    tr_s.append((p[ci], p[ci+1]))
  else: 
    raise ValueError("invalid")
  p = swap(p, ci)
  print(p)


# def schedule(f0: ArrayLike, f1: ArrayLike):
n = 100
f0 = np.random.uniform(size=n, low=0, high=1)
f1 = np.random.uniform(size=n, low=0, high=1)

import copy 

# sgn_p = np.array([np.sign(index_of(p, sa) - index_of(p, c)) for c in p])
# sgn_q = np.array([np.sign(index_of(q, sa) - index_of(q, c)) for c in p])
# s = np.array(list(ascii_lowercase[:n]))
# S0 = np.random.choice(s, size=len(s), replace=False)
# S1 = np.random.choice(s, size=len(s), replace=False)
# sa_idx, sb_idx = index_of(p, sa), index_of(p, sb)
# [index_of(p, c) for c in p]
# sgn_sb0 = sum([np.sign(sa_idx - index_of(p, c)) >= 0 and np.sign(sb_idx - index_of(p, c)) for c in p])

# # np.sign(index_of(p, sa) - index_of(q, sa))
# num_local_concordant = lambda sym, p:

# ## local_concordant 
# def is_cc(a,b,p,q):
#   sa_sgn = np.sign(index_of(p, b) - index_of(p, a))
#   sb_sgn = np.sign(index_of(q, b) - index_of(q, a))
#   return(sa_sgn == sb_sgn)
# sum([pair_is_concordant(1,c,r,q) for c in p])


# sum([np.sign(sa_idx - index_of(p, c)) >= 0 and np.sign(sb_idx - index_of(p, c)) for c in p])

# sgn_sa0 = np.array([np.sign(sa_idx - index_of(p, c)) for c in p])
# sgn_sa1 = np.array([np.sign((sa_idx+1) - index_of(p, c)) for c in p])

# sum(sgn_sa0 >= 0)
# sum(sgn_sa1 >= 0)

# sb_idx = index_of(p, sb)
# sgn_sb0 = sum([np.sign(sb_idx - index_of(p, c)) >= 0 and np.sign(sb_idx - index_of(p, c)) for c in p])
# sgn_sb1 = np.array([np.sign((sb_idx+1) - index_of(p, c)) for c in p])

# sum(sgn_sb0 >= 0)
# sum(sgn_sb1 >= 0)

# sum(sgn_sa0 >= 0) + sum(sgn_sb0 >= 0)
# sum(sgn_sa1 >= 0) + sum(sgn_sb1 >= 0)

# sgn_q = np.array([np.sign(index_of(q, sa) - index_of(q, c)) for c in p])
# sum(sgn_p == sgn_q)

# swap(p, index_of(p, sa))
#     # if is_discordant:
#     #   r = swap(p, index_of(sa,p))
#     #   n_cc_u = sum([is_cc(sa,c,p,q) for c in p]) + sum([is_cc(sb,c,p,q) for c in p]) # unswapped
#     #   n_cc_s = sum([is_cc(sa,c,r,q) for c in r]) + sum([is_cc(sb,c,r,q) for c in r]) # swapped
#     #   if n_cc_s > n_cc_u: # if # concordant pairs increases
#     #     int_pt = intersection(line((0, F0[sa]), (1, F1[sa])), line((0, F0[sb]), (1, F1[sb])))
#     #     assert int_pt
#     #     cross_f[i] = np.inf if (int_pt[0] < 0 or int_pt[0] > 1) else int_pt[0]
#     #   else: 
#     #     cross_f[i] = np.inf