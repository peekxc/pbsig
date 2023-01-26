import numpy as np 
import scipy.sparse 
import _persistence as pm
from scipy.sparse import * 
from pbsig.persistence import *
from pbsig.vineyards import move_right
from pbsig.datasets import random_lower_star
from pbsig.simplicial import SimplicialComplex

def test_matrix_addition():
  A = scipy.sparse.random(n=10, m=10, density=0.05).tocsc()
  F = pm.FloatMatrix(A)
  low_entries = [max(A[:,[i]].indices) if A[:,[i]].nnz > 0 else -1 for i in range(A.shape[1])]
  assert [F.low(i) for i in range(F.dim[1])] == low_entries
  F.clear_column(0)
  assert F.low(0) == -1
  F.iadd_scaled_col(0,1,1.0)
  assert F.low(0) == F.low(1) 
  F.swap_cols(2,3)
  assert all(F.as_spmatrix()[:,[2]].indices == A[:,[3]].indices)
  assert all(F.as_spmatrix()[:,[3]].indices == A[:,[2]].indices)

def test_basic_move_right():
  D = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
  R = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
  V = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
  D = scipy.sparse.bmat([[None, D], [np.zeros((3,3)), None]]).tolil()
  R = scipy.sparse.bmat([[None, R], [np.zeros((3,3)), None]]).tolil()
  V = scipy.sparse.bmat([[np.eye(3), None], [None, V]]).tolil()
  assert np.allclose(R.todense(), (D @ V).todense() % 2)
  R, V = R.astype(int), V.astype(int)


def test_move_right():
  for cj in range(100):
    np.random.seed(cj)
    X, K = random_lower_star(15)
    D = boundary_matrix(K)
    V = eye(D.shape[0])
    I = np.arange(0, D.shape[1])
    R, V = pm.phcol(D, V, I)
    assert is_reduced(R)
    assert np.isclose((R - (D @ V)).sum(), 0.0)
    D, R, V = D.astype(int).tolil(), R.astype(int).tolil(), V.astype(int).tolil()

    ## Try random movements that respect the face poset
    S = list(K.values())
    valid_mr = []
    for i,j in combinations(range(len(S)), 2):
      respects_face_poset = all([not(S[i] <= s) for s in S[(i+1):(j+1)]])
      if (respects_face_poset): # comment out to try non-face moves
        valid_mr.append((i,j))
    
    cc = np.argmax(np.diff(np.array(valid_mr), axis=1))
    i, j = valid_mr[cc]
    # print(S[i])
    # print(S[(i+1):(j+1)])
    #plt.spy(R)
    assert is_reduced(move_right(R, V, i, j, copy=True)[0])
    #move_right(R, V, i, j, copy=True)  
    move_right(R, V, i, j, copy=False)  
    #plt.spy(R)

    idx_set = np.arange(len(S))
    rind = np.roll(np.arange(i,j+1), -1)
    bind = np.arange(len(S))
    bind[i:(j+1)] = rind
    S = [S[i] for i in bind]
    K = MutableFiltration(zip(idx_set, S))
    PDP = boundary_matrix(K)  
    # plt.spy(PDP @ V)

    assert is_reduced(R), "R is not reduced"
    assert all([r <= c for r,c in zip(*V.nonzero())]), "V is not upper-triangular"
    assert all(V.diagonal()%2 >= 0), "V is not full rank"
    assert is_reduced((PDP @ V).todense() % 2), "D @ V is not reduced"
    assert np.allclose(((PDP @ V).todense() - R.todense()) % 2, 0), "R = D @ V does not hold"
  

from pbsig.vineyards import *
def test_move_left():
  for cj in range(100):
    np.random.seed(cj)
    X, K = random_lower_star(15)
    D = boundary_matrix(K)
    V = eye(D.shape[0])
    I = np.arange(0, D.shape[1])
    R, V = pm.phcol(D, V, I)
    assert is_reduced(R)
    assert np.isclose((R - (D @ V)).sum(), 0.0)
    D, R, V = D.astype(int).tolil(), R.astype(int).tolil(), V.astype(int).tolil()

    ## Try random movements that respect the face poset
    S = list(K.values())
    valid_ml = []
    for i,j in combinations(range(len(S)), 2):
      respects_face_poset = not(any([s <= S[j] for s in S[i:j]]))
      if (respects_face_poset):
        valid_ml.append((i,j))
    
    cc = np.argmax(np.diff(np.array(valid_ml), axis=1))
    i, j = valid_ml[cc]
    # print(S[j])
    # print(S[i:j])
    # plt.spy(R)
    assert is_reduced(move_left(R, V, j, i, copy=True)[0])
    #move_right(R, V, i, j, copy=True)  
    move_left(R, V, j, i, copy=False)  
    #plt.spy(R)


    P = permutation_matrix(move_left_permutation(i,j, len(K)))
    R = R.todense() % 2
    V = V.todense() % 2 
    PDP = (P @ D @ P.T).todense() % 2
    assert is_reduced(R), "R is not reduced"
    assert all([r <= c for r,c in zip(*V.nonzero())]), "V is not upper-triangular"
    assert all(np.ravel(V.diagonal()%2) >= 0), "V is not full rank"
    assert is_reduced((PDP @ V) % 2), "D @ V is not reduced"
    assert np.allclose(((PDP @ V) - R) % 2, 0), "R = D @ V does not hold"
    print(cj)


def schedule():
  method = "greedy"
  N = 40
  for cc in range(100):
    np.random.seed(cc)
    L_orig = np.random.choice(range(N), size=N, replace=False)
    lis = np.array(longest_subsequence(L_orig))
    com = np.setdiff1d(L_orig, lis)
    L = np.array([-1] + list(L_orig) + [len(L_orig)])

    ## Define successor, predecessor, and position functions
    succ = lambda L, i: L[np.flatnonzero(i < L)[0]] if any(i < L) else N
    pred = lambda L, i: L[np.flatnonzero(L < i)[-1]] if any(L < i) else -1
    pos = lambda L, i: np.flatnonzero(L == i)[0]

    ## Used to generate candidate moves 
    def query_move(symbol, lst, lis):
      i = pos(lst, symbol)
      j = pos(lst, pred(lis, symbol)) ## b
      k = pos(lst, succ(lis, symbol)) ## e
      t = j if i < j else k    ## target index, using nearest heuristic 
      return (i,t)

    ## Generate the schedule
    schedule = []
    while not(is_sorted(L)):
      if method == "nearest":
        d = com[0]
        i,t = query_move(d, L, lis)
      elif method == "random":
        d = np.random.choice(com, size=1).item()
        i,t = query_move(d, L, lis)
      elif method == "greedy": 
        Q = [query_move(s, L, lis) for s in com]
        I = np.array(Q).flatten().astype(np.int32)
        i,t = Q[np.argmin(comb_mod.interval_cost(I))]
      else: 
        assert isinstance(method, Callable)
        Q = [query_move(s, L, lis) for s in com]
        i,t = method(Q)
      
      print(f"L: {L}, LIS: {lis}, complement: {com}, symbol: {L[i]}")
      
      ## Move a symbol in the complement set and update the LIS and L 
      sym = L[i]
      assert sym in com, "Invalid symbol chosen"
      com = np.setdiff1d(com, sym)
      L = permute_cylic_pure(L, min(i,t), max(i,t), right=i < t)
      lis = np.sort(np.append(lis, sym))
      schedule.append((i,t))

    ## Check 
    S = L_orig.copy()
    for i,j in schedule:
      i,j = i-1,j-1
      S = permute_cylic_pure(S, min(i,j), max(i,j), right=i < j)
    assert is_sorted(S)

from pbsig.vineyards import move_schedule, move_left, move_right
def test_scheduled_moves():
  for cj in range(100):
    np.random.seed(cj)
    X, K = random_lower_star(25)
    D = boundary_matrix(K)
    V = eye(D.shape[0])
    I = np.arange(0, D.shape[1])
    R, V = pm.phcol(D, V, I)
    assert is_reduced(R)
    assert np.isclose((R - (D @ V)).sum(), 0.0)
    
    ## New filtration 
    v = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
    fv = X @ v
    L = MutableFiltration(K.values(), f = lambda s: max(fv[s]))

    ## Scheduling sorts an arrangement to the identity, so label K using L
    L_map = { s : i for i,s in enumerate(L.values()) }
    K_perm = np.array([L_map[s] for s in K.values()], dtype=np.int32)

    R = R.astype(int).tolil()
    V = V.astype(int).tolil()
    schedule = move_schedule(K_perm, method="greedy")
    for i,j in schedule:
      if i < j:
        move_right(R, V, i, j)
      else:
        move_left(R, V, i, j)
      assert is_reduced(R), "R is not reduced"
      assert all([r <= c for r,c in zip(*V.nonzero())]), "V is not upper-triangular"
      assert all(np.ravel(V.diagonal()%2) >= 0), "V is not full rank"
    D = boundary_matrix(L).astype(int)
    assert np.allclose(((D @ V).todense() % 2) - (R.todense() % 2), 0)
    print(cj)

def test_schedule_coarsening():
  from pbsig.vineyards import move_schedule
  n = 150
  p = np.random.choice(range(n), size=n, replace=False)
  SS = [move_schedule(p, coarsen=c, verbose=False).shape[0] for c in np.linspace(0, 1, 30)]
  assert SS[0] == n - 1, "Minimum coarseness should in theory require n-1 moves"
  assert all(np.diff(SS) <= 0), "Move schedules at different coarseness levels not monotone"

def test_ls_moves():
  from pbsig.vineyards import update_lower_star
  from pbsig.vineyards import move_stats
  X, K = random_lower_star(15)
  D = boundary_matrix(K)
  V = eye(D.shape[0])
  I = np.arange(0, D.shape[1])
  R, V = pm.phcol(D, V, I)
  assert is_reduced(R) and np.isclose((R - (D @ V)).sum(), 0.0)

  fv = X @ np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
  R = R.astype(int).tolil()
  V = V.astype(int).tolil()
  assert update_lower_star(K, R, V, lambda s: max(fv[s])) is None


def test_vineyards():
  from pbsig.datasets import random_lower_star
  from pbsig.vineyards import transpose_rv, linear_homotopy
  X, K = random_lower_star(15)
  D = boundary_matrix(K)
  V = eye(D.shape[0])
  I = np.arange(0, D.shape[1])
  R, V = pm.phcol(D, V, I)
  assert is_reduced(R) and np.isclose((R - (D @ V)).sum(), 0.0)

  fv = X @ np.array([-1, 0])
  L = MutableFiltration(K.values(), f=lambda s: max(fv[s]))

  KD = { v:k for k,v in K.items() }
  LD = { v:k for k,v in L.items() }
  tr_schedule, dom = linear_homotopy(KD, LD, plot=False)

  R = R.astype(int).tolil()
  V = V.astype(int).tolil()
  assert is_reduced(R)
  for cc, s in enumerate(transpose_rv(R, V, tr_schedule)):
    assert is_reduced(R), "R is not reduced"
    assert all([r <= c for r,c in zip(*V.nonzero())]), "V is not upper-triangular"
    assert all(np.ravel(V.diagonal()%2) >= 0), "V is not full rank"
  status = [s for s in transpose_rv(R, V, tr_schedule)]
  

  #def transpose_rv(R: lil_array, V: lil_array, I: Iterable):

