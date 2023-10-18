from typing import *
import numpy as np 
from scipy.sparse import csc_array, coo_array
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve, lsqr, splu
from scipy.optimize import linprog
from itertools import islice
from scipy.sparse.csgraph import structural_rank
from itertools import product
from .utility import progressbar
from .linalg import rank_bound
from numbers import Number

# Miller-Rabin primality test
def is_prime(n: int):
  return False if n % 2 == 0 and n > 2 else all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

def gen_primes_above(m: int, n: int):
  """Generate _n_ primes above _m_."""
  m,n = int(m),int(n)
  primes = []
  while len(primes) < n:
    prime_candidates = np.fromiter(islice(filter(is_prime, range(m, 2*m - 2)), 100), dtype=int) ## Bertrand's postulate
    primes.extend(np.unique(prime_candidates))
    m = primes[-1]+1
  return np.array(primes)


## Universal Hash Function (Linear Congruential Generator)
## https://en.wikipedia.org/wiki/Linear_congruential_generator
## NOTE: If c = 1, then the prime will be fixed which isn't good for hashing as multiple hash functions
## will be be more likely to be linearly independent.  
class LCG:
  """Produces a (random) Universal Hash Function. 
  
  Parameters: 
    m = universe size 
    c = number of primes larger than m to sample from. Defaults to 100. 
  
  Returns: 
    a random LCG hash function (x, params) which hashes integer inputs. Supply params=True to see the parameters.
  """
  def __init__(self, primes: Iterable):
    self.primes = np.fromiter(iter(primes), dtype=np.int64)

  def randomize(self, m: int):
    self.m = m
    self.p = np.random.choice(self.primes)      ## Random prime from first |c| primes after m 
    self.a = max([1,np.random.choice(self.p)])
    self.b = np.random.choice(self.p)         ## Multiplier/increment (a,b) 
  
  def __call__(self, x: int, params: bool = False):
    return ((self.a*x + self.b) % self.p) % self.m if not params else (self.a,self.b,self.p,self.m)

import random
class IntSaltHash:
  def __init__(self):
    self.salt = []

  def randomize(self, m: int):
    self.m = m
    self.salt = []

  def __call__(self, key: str):
    key = str(key)
    while len(self.salt) < len(key): # add more salt as necessary
      self.salt.append(random.randint(1, self.m - 1))
    return sum(self.salt[i] * ord(c) for i, c in enumerate(key)) % self.m


def perfect_hash(S: Iterable[int], output: str = "function", solver: str = "linprog", k_min: int = 2, k_max: int = 15, n_tries: int = 100, n_prime: int = 100):
  """Defines a perfect minimal hash.

  Defines a perfect minimal hash function mapping the 
  
  Parameters:
    S: the set of keys to hash. Must be an iterable of integers. 
    output: desired output. Can be either 'function', 'expression', 
    k_min: minimum number of LCGs to compose into the final hash function
    k_max: maximum number of LCGs to compose into the final hash function
    n_tries: number of attempts to solve the linear system to produce the pmh.  
    n_prime: number of primes to sample from 
  """
  assert k_min <= k_max
  assert solver == "linprog" or solver == "lu" or solver == "lsqr"
  S = np.fromiter(iter(S), dtype=np.uint64)
  N = len(S)
  b = max(1, int(np.ceil(n_tries/(max(1, abs(k_max-k_min)))))) # attempts per iteration
  success = False 
  n_solves = 0
  n_attempts = 0
  H = None
  primes = gen_primes_above(max(S), n_prime)
  f = LCG(primes)
  HF = [LCG(primes) for _ in range(k_max)]
  from copy import deepcopy

  #print(f"k_min: {k_min}, k_max:{k_max}, block size: {b}")
  for k, _ in progressbar(product(range(k_min, k_max+1), range(b)), n_tries):
    n_attempts += 1
    ## Sample uniformly random universal hash functions
    # HF = [random_hash_function(N, primes) for i in range(k)]
    for i in range(k):
      HF[i].randomize(N)

    ## Build the hash matrix 
    I, J = [], []
    for i,s in enumerate(S):
      I += [i] * k
      J += [h(s) for h in HF[:k]]
    H = coo_array(([1]*(N*k), (I,J)), shape=(N, N))
    ## TODO: make non-data int32 type for newest scipy
    
    ## If the structural rank isn't full rank, no way its feasible
    ## Otherwise, Try to find a pmf with the chosen solver
    if structural_rank(H) < N:
      continue
    elif rank_bound(H @ H.T, upper=True) < N:
      print(f"{rank_bound(H)} < {N}")
      continue 
    else: 
      n_solves += 1
      if solver == "linprog":
        H = H.tocsr()
        res = linprog(c=np.ones(N), A_eq=H, b_eq=np.arange(N).astype(int), bounds=(None, None)) # , options={'maxiter':1})
        if res.success:
          # print("Success")
          success = True
          g = res.x
          break
      elif solver == "lu":
        g = spsolve(H.tocsc(), np.arange(N))
        if all([int(round(sum([g[int(h(x))] for h in HF[:k]]))) == i for i, x in enumerate(S)]):
          # print("Success")
          success = True 
          break
      elif solver == "lsqr":
        from scipy.sparse.linalg import cg, cgs, qmr, lsqr
        _ = lsqr(H.tocsr(), np.arange(N), atol=1e-15)
        g = _[0]
        # print(f"{H @ g}")
        # print([int(round(sum([g[int(h(x))] for h in HF[:k]])))-i for i,x in enumerate(S)])
        if all([int(round(sum([g[int(h(x))] for h in HF[:k]]))) == i for i, x in enumerate(S)]):
          # print("Success")
          success = True 
          break
    if success: break

  ## Throw if couldn't make one
  if not success: 
    import warnings 
    #print(f"Failure on n={N} set size with n_hash: {k}, structural rank: {structural_rank(H)} ({n_attempts} attempts)")
    warnings.warn(f"Unable to create PMH on n={N} sized set using #{n_solves} solves ({n_attempts} attempts, structural rank={structural_rank(H)}).")
    return None
  
  ## Otherwise, choose the return type 
  if output == "function": 
    def _perfect_hash(x: int) -> int:
      return int(round(sum([g[h(x)] for h in HF])))
    return _perfect_hash
  elif output == "expression":
    r,c = H.nonzero()
    expr = '+'.join(["g[(({0}*x+{1})%{2})%{3}]".format(*h(0, True)) for h in HF])
    return g, expr
  else: 
    raise ValueError("Unknown output type")


## Modified from: https://github.com/toymachine/chdb/blob/3e258c34832acf98cac9c8e2b5d75421cb4add12/perfect_hash.py and https://github.com/ilanschnell/perfect-hash
## Copyright (c) 2019 - 2021, Ilan Schnell
class Graph():
  def __init__(self):
    self.adj = defaultdict(list)

  def reset(self):
    self.adj = defaultdict(list)

  ## Connect vertices using hash function with edge w/ weight = (0, 1, ..., N-1)
  def connect_all(self, keys: Iterable, f1: Callable, f2: Callable):
    for hashval, key in enumerate(keys):
      v0, v1 = f1(key), f2(key)
      self.adj[v0].append((v1, hashval))
      self.adj[v1].append((v0, hashval))

  def assign_values(self, N: int) -> bool:
    """
    Try to assign the vertex values, such that, for each edge, you can add the values for the two vertices involved and get the desired
    value for that edge, i.e. the desired hash key. This will fail when the graph is cyclic.

    This is done by a Depth-First Search of the graph.  If the search finds a vertex that was visited before, there's a loop and False is 
    returned immediately, i.e. the assignment is terminated. On success (when the graph is acyclic) True is returned.
    """
    self.vertex_values = -np.ones(N, dtype=np.int32) 
    visited = np.zeros(N, dtype=bool)
    
    # Loop over all vertices, taking unvisited ones as roots.
    for root in range(N):
      if visited[root]: continue
      self.vertex_values[root] = 0  # set arbitrarily to zero
      to_visit = [(None, root)]     # (parent, vertex)
      while len(to_visit) > 0:
        parent, vertex = to_visit.pop()
        visited[vertex] = True

        ## Loop over adjacent vertices, but skip the vertex we arrived here from the first time it is encountered.
        skip = True
        for neighbor, edge_value in self.adj[vertex]:
          if skip and neighbor == parent:
            skip = False
            continue
          if visited[neighbor]: return False # graph is cyclic! 
          to_visit.append((vertex, neighbor))

          ## Assignment step: set vertex value to the desired edge value, minus the value of the vertex we came here from.
          # print(f"Assigning: g[{neighbor}] = ({edge_value} - g[{vertex}]) % {N}  == {(edge_value - self.vertex_values[vertex]) % N}")
          self.vertex_values[neighbor] = (edge_value - self.vertex_values[vertex]) % N
    
    return True # success: graph is acyclic so all values are now assigned.

from collections import defaultdict
def perfect_hash_dag(keys: Iterable[int], output: str = "function", use_lcg: bool = True, mult_max: float = 2.0, n_tries: int = 100, n_prime: int = 100, progress: bool = False):
  N = len(keys)
  G = Graph()
  n_attempts = 0 
  success = False 
  step_sz = (np.ceil(mult_max*N)-N)/n_tries
  if use_lcg:
    primes = gen_primes_above(max(keys), n_prime)
    f1 = LCG(primes)
    f2 = LCG(primes)
  else: 
    f1 = IntSaltHash()
    f2 = IntSaltHash()
  it = progressbar(range(n_tries), count=n_tries) if progress else range(n_tries)
  for i in it:
    NG = int(N + i*step_sz)
    f1.randomize(NG)
    f2.randomize(NG)

    ## Connect vertices using hash function then check for valid assignments
    G.reset()
    G.connect_all(keys, f1, f2)
    if G.assign_values(NG):
      success = True 
      break
    else: 
      n_attempts += 1

  if success:
    g = G.vertex_values.astype(int)
    assert all([hashval == int((g[f1(key)]+g[f2(key)]) % NG) for hashval, key in enumerate(keys)])
    if output == "expression":
      expr = '(' + '+'.join(["g[(({0}*x+{1})%{2})%{3}]".format(*h(0, True)) for h in [f1,f2]]) + ')' + f"%{NG}"
      return g, expr, G
    else:
      def _perfect_hash(x: int) -> int:
        return int(sum([g[h(x)] for h in [f1,f2]]) % NG)
      return _perfect_hash, G
  else: 
    g = G.vertex_values.astype(int)
    n_fails = sum([hashval != int((g[f1(key)]+g[f2(key)]) % NG) for hashval, key in enumerate(keys)])
    import warnings
    warnings.warn(f"Unable to create PMH on n={N} sized set using #{n_attempts} DAG attempts ({n_fails} collisions in final attempt).")
    return None 

# from scipy.sparse import coo_array 
# E = np.unique(np.array([(f1(key), f2(key)) for key in keys if f1(key) != f2(key)], dtype=np.uint16), axis=1)
# if (len(E) >= NG):
#   continue
# G = coo_array((np.arange(len(E)), tuple(E.T))) 
# depth_first_order(G, 0)