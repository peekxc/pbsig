from typing import *
import numpy as np
from scipy.sparse import *
from pbsig.persistence import *
from pbsig.simplicial import *
from pbsig.vineyards import linear_homotopy, transpose_rv
from pbsig.datasets import *
import _persistence as pm


def test_vineyards():
  X, K = random_lower_star(15)
  D, V = boundary_matrix(K), eye(len(K))
  R, V = pm.phcol(D, V, range(len(K)))
  R = R.astype(int).tolil()
  V = V.astype(int).tolil()
  for t in (np.linspace(0, 2*np.pi, 12)+np.pi/2):
    fv = X @ np.array([np.cos(t), np.sin(t)])
    L = MutableFiltration(K.values(), f=lambda s: max(fv[s]))
    S, _ = linear_homotopy(K, L)
    for s in transpose_rv(R, V, S):
      assert is_reduced(R), "R is not reduced"
      assert all([r <= c for r,c in zip(*V.nonzero())]), "V is not upper-triangular"
      assert all(np.ravel(V.diagonal()%2) >= 0), "V is not full rank"

from pbsig.vineyards import *
from pbsig.vineyards import move_stats, vineyards_stats
def test_vineyards_api():
  X, K = random_lower_star(15)
  D, V = boundary_matrix(K), eye(len(K))
  R, V = pm.phcol(D, V, range(len(K)))
  R = R.astype(int).tolil()
  V = V.astype(int).tolil()
  vineyards_stats(reset = True)
  VS = []
  for theta in np.linspace(0, 2*np.pi, 12)+np.pi:
    v = np.array([np.cos(theta), np.sin(theta)])
    fv = X @ v
    update_lower_star(K, R, V, f=lambda s: max(fv[s]), vines=True)
    VS.append(vineyards_stats())

def test_vineyards_cost():
  for seed in range(100):
    np.random.seed(seed)
    X, K = random_lower_star(25)
    pairs = list(combinations(range(len(K)), 2))
    ind = np.random.choice(range(len(pairs)), size=150)
    for cc in ind:
      i,j = pairs[cc]
      D, V = boundary_matrix(K), eye(len(K))
      R, V = pm.phcol(D, V, range(len(K)))
      R = R.astype(int).tolil()
      V = V.astype(int).tolil()
      vineyards_stats(reset = True)
      I = np.arange(i, j)
      status = list(transpose_rv(R, V, I))
      # print(vineyards_stats())
      
      D, V = boundary_matrix(K), eye(len(K))
      R, V = pm.phcol(D, V, range(len(K)))
      R = R.astype(int).tolil()
      V = V.astype(int).tolil()
      move_stats(reset=True)
      move_right(R, V, i, j)
      # print(move_stats())
      assert move_stats()['n_cols_right'] <= vineyards_stats()['n_cols']
      if abs(i - j) > 1: 
        assert move_stats()['n_cols_right'] <= int(vineyards_stats()['n_cols']/2)
    print(seed)

def test_discordant_pairs():
  from itertools import combinations
  from math import comb
  # def pair_is_discordant(a: int, b: int, p: Sequence, q: Sequence) -> bool:
  #   ap, bp = p.index(a), p.index(b)
  #   aq, bq = q.index(a), q.index(b)
  #   return(False if ap == aq and bp == bq else np.sign(ap - bp) == -np.sign(aq - bq))
  n = 15
  p = list(np.random.choice(range(n), size=n, replace=False))
  q = list(np.random.choice(range(n), size=n, replace=False))
  
  ## Test logical understanding of inverse permutation 
  p_inv = np.argsort(p)
  q_inv = np.argsort(q)
  concordant = lambda i,j: np.sign(p_inv[i] - p_inv[j]) == np.sign(q_inv[i] - q_inv[j])
  assert all([p.index(i) == p_inv[i] for i in range(n)])
  assert all([q.index(i) == q_inv[i] for i in range(n)])

  ## Kendall tau tests 
  kendall_dist = inversion_dist(p,q)
  for i,j in combinations(range(n), 2):
    assert not(concordant(i,j)) == pair_is_discordant(i,j,p,q)
  n_concord = sum([concordant(i,j) for i, j in combinations(range(n), 2)])
  n_discord = sum([not(concordant(i,j)) for i, j in combinations(range(n), 2)])
  assert (comb(n,2) - n_concord) == n_discord
  assert kendall_dist == n_discord

  ## Update p_inv, q_inv under transpositions
  p_inv = np.argsort(p)
  for i in np.random.choice(range(n-1), size=100):
    p_inv[p[i]] += 1
    p_inv[p[i+1]] -= 1
    p[i], p[i+1] = p[i+1], p[i]
    assert all([p.index(x) == p_inv[x] for x in range(n)])