import numpy as np 
from pbsig.combinatorial import * 
from itertools import combinations

def test_rank2():
  s = np.array([rank_C2(i, j, 5) for i,j in combinations([0,1,2,3,4], 2)])
  assert np.allclose(s, np.arange(len(s)))


def test_boundary():

  n, d = 10, 3
  S = [0,1,2,3]
  N = rank_comb(S, k=d+1, n=n)

  ## Lemma 4.1 
  # from math import comb 
  # i_choose_k = [comb(i, d+1) for i in range(n) if comb(i, d+1) <= N]
  # assert len(i_choose_k) == max(S)

  ## Boundary 
  k = d+1 # down to 0
  r = rank_comb([4,5,8], k=3, n=n)
  comb(4,1) + comb(5,2) + comb(8,3)
  # for i, v in enumerate(S)


