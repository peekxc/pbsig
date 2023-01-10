import numpy as np 
from pbsig.combinatorial import * 
from itertools import combinations

def test_rank2():
  s = np.array([rank_C2(i, j, 5) for i,j in combinations([0,1,2,3,4], 2)])
  assert np.allclose(s, np.arange(len(s)))


def test_fast_rank():
  from math import comb
  import numpy as np 
  from itertools import combinations
  from pbsig.combinatorial import comb_mod

  n = 30
  for k in range(2, 5):
    k_ranks = np.sort(np.random.choice(comb(n,k), size=100, replace=False)).astype(np.int32)
    k_combs = [c for i,c in enumerate(combinations(range(n), k)) if i in k_ranks]

    ## Test ranking
    k_combs_vec = np.ravel(k_combs).astype(np.int32)
    k_ranks_test = comb_mod.rank_combs(k_combs_vec, n, k)
    assert all((k_ranks - k_ranks_test) == 0)

    ## Test unranking 
    k_combs_test = comb_mod.unrank_combs(k_ranks, n, k)
    assert all((k_combs_test - k_combs_test) == 0)

  ## Test boundaries
  from pbsig.simplicial import SimplicialComplex
  S = SimplicialComplex([[0,1,2,3,4,5]])
  n, k = S.shape[0], 3
  arr = lambda g: np.array(list(g)).astype(np.int32)
  Q = arr(S.faces(k-1))
  assert all(comb_mod.rank_combs(Q.reshape(-1), n, k)-np.arange(0, S.shape[k-1]) == 0)

  ## Get boundary ranks
  boundary_ranks_test = [comb_mod.rank_combs(arr(s.boundary()).reshape(-1), n, k-1) for s in S.faces(k-1)]
  boundary_ranks_true = [rank_combs(s.boundary(), k-1, n) for s in S.faces(k-1)]
  assert all((arr(boundary_ranks_test).reshape(-1) - arr(boundary_ranks_true).reshape(-1)) == 0)




def test_boundary():
  import numpy as np 
  from itertools import combinations

  from pbsig.combinatorial import comb_mod
  a = np.array([0,1,0,2], dtype=np.int32)
  b = np.array([0,1,2, 0,1,3, 0,2,8], dtype=np.int32)
  
  br = comb_mod.rank_combs(b, 10, 3)
  bu = comb_mod.unrank_combs(br, 10, 3) # unranked 
  assert all(b == bu)

  from math import comb 
  n, k = 30, 3
  r = np.sort(np.random.choice(comb(n,k), 100, replace=False)) # must be sorted
  bu = comb_mod.unrank_combs(r, n, k)

  C = combinations(range(n), k)
  bc = np.array([c for i,c in enumerate(C) if i in r]).astype(np.int32)
  assert all((bc.reshape(-1) - bu) == 0)

  from pbsig.combinatorial import comb_mod
  comb_mod.boundary_ranks(0, 10, 2)
  comb_mod.boundary_ranks(5, 10, 3)


  s = np.array([0,1,7], dtype=np.int32)
  comb_mod.rank_combs(s, 10, 3) # 5 
  comb_mod.boundary_ranks(5, 10, 3) # 0,6,14
  fr = np.array([0,6,14], dtype=np.int32)
  comb_mod.unrank_combs(fr, 10, 2) # 0, 1, 0, 7, 1, 7

  [comb_mod.boundary_ranks(r, L.nv, 3) for r in L.qr]
  
  from pbsig.combinatorial import unrank_combs 
  from array import array
  p_ranks = array('I')
  q_simplices = unrank_combs(L.qr, k=3, n=L.nv)
  for qs in q_simplices:
    pr = rank_combs(combinations(qs, 2), k=2, n=L.nv)
    p_ranks.extend(pr)
  p_ranks = np.unique(np.sort(p_ranks))
  np.sort(list(L.index_map.keys()))
  # n, d = 10, 3
  # S = [0,1,2,3]
  # N = rank_comb(S, k=d+1, n=n)

  ## Lemma 4.1 
  # from math import comb 
  # i_choose_k = [comb(i, d+1) for i in range(n) if comb(i, d+1) <= N]
  # assert len(i_choose_k) == max(S)

  ## Boundary 
  k = d+1 # down to 0
  r = rank_comb([4,5,8], k=3, n=n)
  comb(4,1) + comb(5,2) + comb(8,3)

  # given S 


  # for i, v in enumerate(S)


