import numpy as np
from pbsig.vineyards import spearman_footrule_dist, move_schedule

p = [4,2,7,1,8,6,3,5,0] # hand-picked LIS [1,3,5]
A = { el : el-p.index(el) for el in p }

# sum(np.abs(list(A.values())))
P = [                  # LIS: 
  [4,2,7,1,8,6,3,5,0], # 1,3,5
  [4,2,1,8,6,3,5,7,0], # 1,3,5,7
  [4,2,0,1,8,6,3,5,7], # 0,1,3,5,7
  [4,2,0,1,6,3,5,7,8], # 0,1,3,5,7,8
  [2,0,1,6,3,4,5,7,8], # 0,1,3,4,5,7,8
  [2,0,1,3,4,5,6,7,8], # 0,1,3,4,5,6,7,8
  [0,1,2,3,4,5,6,7,8]  # 0,1,2,3,4,5,6,7,8
]
for p in P:
  # print(spearman_footrule_dist(p))
  for el in p: A[el] = el-p.index(el) 
  disp = [A[i] for i in range(9)]
  assert np.sum(np.abs(disp)) == spearman_footrule_dist(p)
  # print('|'.join([f"+{i}" if i >= 0 else f"{i}" for i in disp]))
  print('|' + '|'.join([f"-{abs(A[i])}" if A[i] > 0 else f"+{abs(A[i])}" for i in p]) + '|')

p = P[5]
lis = np.array([0,1,3,4,5,6,7])
m = len(p)
d = m - len(lis)
com = np.setdiff1d(p, lis)
L = np.array([-1] + list(p) + [m])

## Define successor, predecessor, and position functions
succ = lambda L, i: L[np.flatnonzero(i < L)[0]] if any(i < L) else m
pred = lambda L, i: L[np.flatnonzero(L < i)[-1]] if any(L < i) else -1
pos = lambda L, i: np.flatnonzero(L == i)[0]

## Used to generate candidate moves 
def query_move(symbol, lst, lis):
  i = pos(lst, symbol)
  j = pos(lst, pred(lis, symbol)) ## b
  k = pos(lst, succ(lis, symbol)) ## e
  #_MOVE_STATS["WHAT"] = [lis, lst, pos, pred, succ, symbol]
  if j > k:
      f"predecessor {j} of {symbol} is not behind successor {k} of {symbol} in {lis}"
  assert j <= k, f"predecessor {j} of {symbol} is not behind successor {k} of {symbol} in {lis}"
  if i < j:   # s is behind predecessor
    t = max(j, 0)
  elif k < i: # s is ahead of successor
    t = min(k, m)
  else: 
    # s is actually in the lis, which only happens in coarsening case
    # don't move it all
    assert j < i and i < k, "invalid combination" 
    t = i
  #t = j if i < j else k    ## target index, using nearest heuristic 
  return (i,t)

from pbsig.vineyards import permute_cylic_pure
identity = np.arange(len(p))
moves = [query_move(el, L, lis) for el in p]
for i,j in moves: 
  pn = permute_cylic_pure(L, min(i,j), max(i,j), right=i < j)
  print(spearman_footrule_dist(pn[1:-1], identity))


# r1 = [(0,6), (1,3), (2,7), (3,3), (4,8), (5,7), (6,6), (7,7), (8,3)]
# r2 = [(0,5), (1,2), (2,2), (3,7), (4,6), (5,5), (6,6,), (7,7), (8,2)]
# r3 = 

# com = np.setdiff1d(com, sym)
# L = permute_cylic_pure(L, min(i,t), max(i,t), right=i < t)
# lis = np.sort(np.append(lis, sym))
# [spearman_footrule_dist(permute_cylic_pure(p, i, j), np.arange(len(p))) for i,j in r2]



# [-8, -2, 1, -3, 4, -2, 1, 5, 4]
# [-8, -1, 1, -2, 4, -1, 2, 0, 5]
# [-2, -2, 1, -3, 4, -2, 1, -1, 4]
# [-2, -2, 1, -2, 4, -1, 2, 0, 0]
# [-1, -1, 2, -1, -1, -1, 3, 0, 0]
# [-1, -1, 2, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 0]
# spearman_footrule_dist([4,2,0,1,8,6,3,5,7])


