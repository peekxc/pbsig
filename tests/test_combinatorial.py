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



def test_enumerate_lis(x: Sequence[int]):
  
  x = np.random.choice(np.arange(10), size=10, replace=False)
  # x = np.array([0, 4, 1, 5, 3, 2])
  #x = np.array([1, 0, 4, 5, 2, 3])

  assert isinstance(x, Sequence) and all(np.sort(x) == np.arange(len(x)))
  #if len(x) <= 1: yield x; return
  pred = lambda y,i: max(y[y < i]) if any(y < i) else None
  succ = lambda y,i: min(y[y > i]) if any(y > i) else None
  E = np.array([], dtype=int) # T data structure
  L = np.zeros(len(x), dtype=int) # L[i] := length of LIS ending in i

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

  ## Enumerate LIS
  L = L.astype(int)
  L1 = [x[max(np.flatnonzero(L[:j] == l))] if any(L[:j] == l) else None for j,l in enumerate(L) ] # =
  L2 = [x[max(np.flatnonzero(L[:j] == l-1))] if any(L[:j] == l-1) else None for j,l in enumerate(L) ]  # -1
  
  ## Revert order ?
  # L = L[Q]
  # L1 = np.array(L1)[Q]
  # L2 = np.array(L2)[Q]
  ## Stage 2: obtain the initial LIS
  # index = np.argmax(L)
  # S = np.zeros(int(L[index])).astype(int)
  # S[-1] = index
  # j = L[index] - 1
  # for i in reversed(np.arange(0, index+1)):
  #   if L[i] == j:
  #     S[j] = int(i)
  #     j -= 1

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
      print(out)
      return # I think this is needed
    else:
      enum_lis(z1, out)
    while z1 is not None and L1[z1] is not None:
      enum_lis(L1[z1], out)
      z1 = L2[z1]
  out = np.array([x[max(np.flatnonzero(L == i))] for i in range(1, k+1)])
  z = out[-1]
  for z in reversed(x): enum_lis(z, out)
  #return results

def test_lis_enum():
  L = all_longest_subseq(np.random.choice(np.arange(150), size=150, replace=False))
  assert all([is_sorted(l) for l in L])