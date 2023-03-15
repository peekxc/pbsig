from typing import *
import numpy as np 
from scipy.sparse import csc_array, coo_array
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve, lsqr, splu
from scipy.optimize import linprog
from itertools import islice
from scipy.sparse.csgraph import structural_rank

# Miller-Rabin primality test
def is_prime(n: int):
  return False if n % 2 == 0 and n > 2 else all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))


# primes = np.fromiter(filter(is_prime, np.arange(1000)), int)

## Universal Hash Function (Linear Congruential Generator)
## https://en.wikipedia.org/wiki/Linear_congruential_generator
## NOTE: If c = 1, then the prime will be fixed which isn't good for hashing apparently. 
## In particular, H will likely not achieve full rank. 
def random_hash_function(m: int, c: int = 100):
  """Produces a (random) Universal Hash Function 
  
  Parameters: 
    m := universe size 
  """
  np.random.seed()

  ## Choose random prime from first |c| primes after m 
  primes = list(islice(filter(is_prime, range(m, 2*m - 2)), c)) ## Bertrand's postulate
  p = np.random.choice(primes)

  ## Choose multiplier/increment (a,b) 
  a,b = np.random.choice(range(1, p)), np.random.choice(range(p))

  def __hash_fun__(x: int, params: bool = False):
    return ((a*x + b) % p) % m if not params else (a,b,p,m)
  return __hash_fun__

def perfect_hash(S: Iterable[int], output: str = "function", solver: str = "linprog", k_min: int = 2, k_max: int = 15, n_tries: int = 100, n_prime: int = 10):
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
  assert solver == "linprog" or solver == "spsolve"
  S = np.fromiter(iter(S), dtype=np.int32)
  N = len(S)
  b = max(1, int(np.ceil(n_tries/(max(1, abs(k_max-k_min)))))) # attempts per iteration
  success = False 
  n_solves = 0
  n_attempts = 0
  H = None
  #print(f"k_min: {k_min}, k_max:{k_max}, block size: {b}")
  for k in range(k_min, k_max+1):
    for cc in range(b):
      n_attempts += 1
      ## Sample uniformly random universal hash functions
      HF = [random_hash_function(N) for i in range(k)]

      ## Build the hash matrix 
      I, J = [], []
      for i,s in enumerate(S):
        I += [i] * len(HF)
        J += [h(s) for h in HF]
      H = coo_array(([1]*(N*len(HF)), (I,J)), shape=(N, N))
      
      ## If the structural rank isn't full rank, no way its feasible
      ## Otherwise, Try to find a pmf with the chosen solver
      if structural_rank(H) < N:
        continue
      else: 
        n_solves += 1
        if solver == "linprog":
          res = linprog(c=np.ones(N), A_eq=H, b_eq=np.arange(N).astype(int), bounds=(None, None))
          if res.success:
            # print("Success")
            success = True
            g = res.x
            break
        else:
          g = spsolve(H.tocsc(), np.arange(N))
          if all([int(round(sum([g[h(x)] for h in HF]))) == i for i, x in enumerate(S)]):
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



  ## Compute determinant and rank
  # d = np.linalg.det(H.todense())
  # r = np.linalg.matrix_rank(H.todense())


