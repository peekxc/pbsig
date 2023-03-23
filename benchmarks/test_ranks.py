def find_k(r, m):
  if (r == 0 or m == 0): 
    return m-1
  elif (m == 1):
    return r-1
  elif (m == 2): 
    return ceil(sqrt(1 + 8*r)/2)
  elif (m == 3):
    return ceil(pow(6*r, 1/3))
  else: 
    return m - 1 

from splex.combinatorial import rank_colex, unrank_colex
import numpy as np 
from math import comb
ranks = np.random.choice(range(comb(500, 3)), size=200, replace=False)
n, k = 500, 3
for r in ranks:
  s = unrank_colex(r, k=3)
  r = rank_colex(s)
  print(f"Unranking {r} => {s}")
  c = [0]*k
  for i in reversed(range(1, k+1)):
    m = i
    assert find_k(r,m) <= r
    #print(f"Starting search {m} for r={r} in range [{m},{n}] (find_k={find_k(r,m)})")
    while r >= comb(m,i):
      #print(f"{r} >= C({m},{i})={comb(m,i)} so searching {m+1}")
      m += 1
    #print(f"Found {m} for r: {r} (comb: {comb(m,i)} >= {r})")
    assert find_k(r,m) <= m
    c[i-1] = m-1
    r -= comb(m-1,i)
  assert tuple(c) == s

rank_colex([0,4,5])

unrank_colex(16, 3)
