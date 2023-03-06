import numpy as np 
from scipy.sparse import csc_array, coo_array
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve, lsqr, splu
#from pbsig.utility import rank_combs

# Miller-Rabin primality test
is_prime = np.vectorize(lambda n: False if n % 2 == 0 and n > 2 else all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2)))
primes = np.fromiter(filter(is_prime, np.arange(1000)), int)

n = 100 # alphabet size
S = np.random.choice(range(n), size=50, replace=False) # alphabet size
N = len(S)  # size of index set to map to

## Optimal hash
def hash0(k):
  return k

# Thomas Wang hash: http://burtleburtle.net/bob/hash/integer.html
def hash1(a, r):
  a -= (a<<6)
  a ^= (a>>17)
  a -= (a<<9)
  a ^= (a<<4)
  a -= (a<<3)
  a ^= (a<<10)
  a ^= (a>>15)
  return a % r

## Java HashMap 
def hash2(x):
  x ^= (x >> 20) ^ (x >> 12)
  return x ^ (x >> 7) ^ (x >> 4)

## Burtle-Bee's "half-avalanche" 
def hash3(a):
  a = (a+0x479ab41d) + (a<<8)
  a = (a^0xe4aa10ce) ^ (a>>5)
  a = (a+0x9942f0a6) - (a<<14)
  a = (a^0x5aedd67d) ^ (a>>3)
  a = (a+0x17bea992) + (a<<7)
  return a

## Burtle-Bee's "full-avalanche"
def hash4(a):
  a = (a+0x7ed55d16) + (a<<12)
  a = (a^0xc761c23c) ^ (a>>19)
  a = (a+0x165667b1) + (a<<5)
  a = (a+0xd3a2646c) ^ (a<<9)
  a = (a+0xfd7046c5) + (a<<3)
  a = (a^0xb55a4f09) ^ (a>>16)
  return a

## Mid-Square method
def hash5(a, N):
  r = int(np.ceil(np.log10(N)))
  L = list(str(a**2))
  piv = int(len(L)/2)
  return int(''.join(L[piv:(piv+r)]))

## Robert Jenkin Hash (https://gist.github.com/badboy/6267743)
def hash6(key):
  key = (key << 15) - key - 1
  key = key ^ (key >> 12)
  key = key + (key << 2)
  key = key ^ (key >> 4)
  key = (key + (key << 3)) + (key << 11)
  key = key ^ (key >> 16)
  return key

## Knuth's multicative hash 
def hash7(key, p = 15):
  assert(p >= 0 and p <= 32)
  knuth = 2654435769
  return (key * knuth) >> (32 - p)

## Universal Hash Function (Linear Congruential Generator)
## https://en.wikipedia.org/wiki/Linear_congruential_generator
## NOTE: If c = 1, then the prime will be fixed which isn't good for hashing apparently. 
## In particular, H will likely not achieve full rank. 
def random_hash_function(m: int, c: int = 100):
  """
  Produces a (random) Universal Hash Function 
  Parameters: 
    m := universe size 
  """
  from itertools import islice
  np.random.seed()
  ## Choose random prime from first |c| primes after m 
  primes = list(islice(filter(is_prime, range(m, 2*m - 2)), c)) ## Bertrand's postulate
  p = np.random.choice(primes)

  ## Choose multiplier/increment (a,b) 
  a,b = np.random.choice(range(1, p)), np.random.choice(range(p))

  def __hash_fun__(x: int, params: bool = False):
    return ((a*x + b) % p) % m if not params else (a,b,p,m)
  return __hash_fun__

from scipy.sparse.csgraph import structural_rank
#hash_function = lambda x: np.random.choice(primes)*(x+np.random.choice(range(N))) % N
 
N_ITER = 100    # number of iterations to try 
N_HASH = 15     # number of random hash functions to use
#N_PRIME = 10    # number of primes to sample from 
for n_hash in range(2, int(N/10)):
  for cc in range(5):
    ## Sample uniformly random universal hash functions
    HF = [random_hash_function(N) for i in range(n_hash)]
    
    ## Build the hash matrix 
    I, J = [], []
    for i,s in enumerate(S):
      I += [i] * len(HF)
      J += [h(s) for h in HF]
    H = coo_array(([1]*(N*len(HF)), (I,J)), shape=(N, N))
    
    ## Compute determinant and rank
    # d = np.linalg.det(H.todense())
    # r = np.linalg.matrix_rank(H.todense())

    res = linprog(c=np.ones(N), A_eq=H, b_eq=np.arange(N).astype(int), bounds=(None, None), integrality=np.ones(N))
    if res.success:
      print("Success")
      break
  # ## Full rank constraint
  # if r == N:
  #   print(f"H is full-rank @ {cc}")
  #   break 

  ## If we insist on a solution comprised of integers, we need H to be unimodular
  # if d in [-1, 1, 0] and r == N:
  #   print(f"H is full-rank and unimodular @ {cc}")
  #   break 

  ## Quick tentative check: is H has full structural rank, 
  ## we may be able to solve for a perfect minimal hash
  # if structural_rank(H) == N and np.linalg.matrix_rank(H.todense()) == N:
  #   print(f"H might be full rank @ {cc}")
  #   break



def perfect_hash(g, HF, integral: bool = False):
  def _hash_key_(x: int) -> int:
    if integral: 
      return round(sum([np.ceil(g[h(x)]) for h in HF]))
    else:
      return round(sum([g[h(x)] for h in HF]))
  return _hash_key_

## The 'closed form' expression of 
'+'.join(["g[(({0}*x+{1})%{2})%{3}]".format(*h(0, True)) for h in HF])

# (lambda x: round(eval('+'.join(["g[(({0}*x+{1})-{2})%{3}]".format(*h(0, True)) for h in HF]))))(0)

from scipy.sparse.linalg import spsolve
g = spsolve(H.tocsc(), np.arange(N))
hf = perfect_hash(g, HF)

assert all(np.array([hf(s) for s in S]) == np.arange(N))

phf = lambda x: int(round(g[((75*x+9)%79)%50]+g[((24*x+15)%61)%50]+g[((58*x+16)%73)%50]))
[phf(s) for s in S]

## Could also solve with linear program 
from scipy.optimize import linprog
res = linprog(c=np.ones(N), A_eq=H, b_eq=np.arange(N), bounds=(None, None)) # integrality=np.ones(N)
print(res.success)

# (H @ res.x.astype(int))+1

## Optimization: val & (4 in hex) <==> val % 4
## Thus, if the modulo is a power of two, we can use right shifting
## For modulo p where p is prime, we cannot use shifting, but perhaps we can use Montgomery reductions 
## https://en.algorithmica.org/hpc/number-theory/montgomery/
## Python implementation: https://asecuritysite.com/rsa/go_mont2
## Speeds up (x * r) mod p, substitute r = 1 for our fixed prime and we have a fast modulo p

# np.linalg.lstsq(HD, np.arange(N))

# np.sum(np.linalg.svd(H.todense(), full_matrices=False, compute_uv=False) >= 1e-9)
# plt.spy(H, markersize=0.5)
# lu, piv = lu_factor(H.todense())
# x = lu_solve((lu, piv), b)
# assert np.all(np.round(H @ x).astype(int) == np.array(range(N)))












#hf = perfect_hash(x, HF)

[sum([x[h(s)] for h in HF]).astype(int) for s in S]
[hf(s) for s in S]

HF[0](S[0])

for cc in range(2, 50):
  R = N # range to modulo the hash functions; must be R > N


  h1 = lambda x: x % R
  h2 = lambda x: 3*(x + 15) % R
  h3 = lambda x: 2*(x + 17) % R
  h4 = lambda x: 55*(x + 13) % R
  h5 = lambda x: 17*(x + 37) % R
  h6 = lambda x: (1779033703 + 2*x) % R
  h7 = lambda x: hash(str(x)) % R 
  HF = [h1, h2, h3, h4, h5, h6, h7]

  I, J = [], []
  for i,s in enumerate(S):
    I += [i] * len(HF)
    J += [h(s) for h in HF]

  ## Build the hash matrix 
  H = coo_array(([1]*(N*len(HF)), (I,J)), shape=(R, R))
  np.linalg.matrix_rank(H.todense())
  
  ## Assert 
  #np.linalg.matrix_rank(H.todense())

  ## LU factorize
  b = np.fromiter(list(range(N))+[0]*(R-N), dtype=int)
  lu, piv = lu_factor(H.todense())
  lu_solve((lu, piv), b)

  
  lu, piv = lu_factor(H)
  x = lu_solve((lu, piv), b)

  res = lsqr(A=H, b=b)
  if np.all((H @ res[0]).astype(int) == np.array(range(N))):
    print("Success")
    break 




(H @ res[0]).astype(int)

hf = perfect_hash(res[0], HF)

[hf(s) for s in S]

len(np.unique(np.floor(H @ res[0]).astype(int)))
len(np.unique(np.ceil(H @ res[0]).astype(int)))


np.linalg.lstsq(A=H.todense(), b=np.fromiter(range(N), dtype=int))
lsqr
# perfect_minimal

scipy.sparse.coo_matrix(()) # data,(i,j)

  # i*2654435761 % 2^32


# uint64_t xorshift(const uint64_t& n,int i){
#   return n^(n>>i);
# }
# uint64_t hash(const uint64_t& n){
#   uint64_t p = 0x5555555555555555ull; // pattern of alternating 0 and 1
#   uint64_t c = 17316035218449499591ull;// random uneven integer constant; 
#   return c*xorshift(p*xorshift(n,32),32);
# }

def hash_uint1(x: int):
  x = ((x >> 16) ^ x) * int(0x45d9f3b)
  x = ((x >> 16) ^ x) * int(0x45d9f3b)
  x = (x >> 16) ^ x
  return x


# https://rosettacode.org/wiki/Montgomery_reduction#Python
import math
class Montgomery:
  BASE = 10
  def __init__(self, modulo):
      self.m = modulo
      self.n = modulo.bit_length()
      self.rrm = (1 << (self.n * 2)) % modulo
  def reduce(self, t):
    a = t
    for i in range(self.n):
      if (a & 1) == 1:
        a = a + self.m
      a = a >> 1
    if a >= self.m:
      a = a - self.m
    return a

mont = Montgomery(17)

x1, x2 = 1,10
r1 = mont.reduce(x1 * mont.rrm)
r2 = mont.reduce(x2 * mont.rrm)
r = 1 << mont.n
prod = mont.reduce(mont.rrm)
base = mont.reduce(x1 * mont.rrm)
exp = x2
while exp.bit_length() > 0:
  if (exp & 1) == 1:
    prod = mont.reduce(prod * base)
  exp = exp >> 1
  base = mont.reduce(base * base)
print(mont.reduce(prod))

# mod53 = MontgomeryReducer(29).pow(10,5)
# this works: https://github.com/nayuki/Nayuki-web-published-code/blob/master/montgomery-reduction-algorithm/montgomery-reducer.py
# mod53.multiply(30, 19)

# 71 % 8




##
from numbers import Number
class Int(Number):
  def __init__(self, x: int = 0):
    self.x = x
  def __hash__(self):
    return 0
  def __repr__(self): 
    return str(self.x)

S = [Int(i) for i in np.random.choice(range(100), size=10)]
h = { s : i for i,s in enumerate(S) }

## Indeed, this does work 
[h[s] for s in S]


from numbers import Integral
from dataclasses import dataclass
import numpy as np 


@dataclass
class Int():
  val: int = 0
  def __int__(self) -> int:
    return self.val

integers = np.random.choice(range(100000*2), size=100000, replace=False)
S = [Int(i) for i in integers]


# import types
# types.MethodType(lambda: 0, Int)
import timeit
Int.__hash__ = lambda self: hash(self.val)

h = { s : i for i,s in enumerate(S) }
timeit.timeit(lambda: [h[s] for s in S], number=100)/100

Int.__hash__ = lambda self: 0
h = { s : i for i,s in enumerate(S) }
timeit.timeit(lambda: [h[s] for s in S], number=100)/100

## Generated from: 
## python perfect_hash.py ../test_keys --hft=2 -o std
# G = [0, 0, 0, 0, 1, 0, 2, 2, 1]
# S1 = [3, 4]
# S2 = [1, 2]
# assert len(S1) == len(S2) == 2
# def hash_f(key, T): return sum(T[i % 2] * ord(c) for i, c in enumerate(key)) % 9
# def perfect_hash(key): return (G[hash_f(key, S1)] + G[hash_f(key, S2)]) % 9

# [(r,i) for (i,r) in enumerate(rank_combs(faces(S, 1)))]



G = [0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 33, 0, 31, 0, 0, 0, 0,
    0, 0, 0, 0, 29, 0, 70, 0, 71, 0, 0, 0, 0, 0, 21, 0, 6, 0, 95, 12, 19,
    75, 117, 82, 0, 48, 0, 0, 0, 10, 29, 0, 34, 119, 0, 29, 0, 51, 0, 37,
    6, 0, 0, 22, 0, 45, 54, 0, 0, 37, 30, 43, 81, 0, 0, 0, 26, 64, 94, 0,
    47, 149, 53, 0, 23, 0, 69, 0, 27, 0, 74, 0, 47, 1, 0, 30, 148, 54, 0,
    44, 60, 0, 67, 58, 0, 0, 73, 62, 118, 0, 0, 0, 2, 109, 88, 137, 0, 0,
    0, 7, 116, 0, 67, 0, 48, 139, 40, 6, 5, 0, 3, 72, 12, 84, 0, 136, 28,
    125, 0, 13, 0, 5, 20, 115, 0, 0, 24, 11, 49, 0, 45, 17, 81, 42]

S1 = [111, 111, 7]
S2 = [64, 59, 99]
assert len(S1) == len(S2) == 3

def hash_f(key, T):
    return sum(T[i % 3] * ord(c) for i, c in enumerate(key)) % 151

def perfect_hash(key):
    return (G[hash_f(key, S1)] + G[hash_f(key, S2)]) % 151

# ============================ Sanity check =============================

K = ["45", "78", "171", "300", "378", "22", "67", "352",
    "8", "17", "38", "57", "107", "9", "13", "123", "256", "303", "381",
    "14", "19", "59", "124", "157", "20", "50", "258", "411", "470", "111",
    "196", "471", "73", "98", "332", "358", "442", "128", "239", "284",
    "443", "64", "100", "114", "334", "360", "181", "416", "475", "164",
    "336", "148", "222", "243", "447", "149", "223", "289", "391", "339",
    "365", "205", "366", "169", "292", "394", "451", "227", "343", "453",
    "319", "425", "485", "252", "297", "298", "457", "323", "429", "402",
    "403", "431", "461"]
assert len(K) == 83

for h, k in enumerate(K):
    assert perfect_hash(k) == h


rank_combs(faces(S, 1))
w = "\n".join([f"{r},{i}" for i,r in enumerate(rank_combs(faces(S, 1)))])

with open("test_keys", "w") as text_file: text_file.write(w)