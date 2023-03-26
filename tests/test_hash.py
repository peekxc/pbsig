import numpy as np 
from pbsig.perfect_hash import perfect_hash

np.random.seed(1234)
n = 5500 # alphabet size
S = np.sort(np.random.choice(range(n), size=1350, replace=False)) # alphabet size

g, expr = perfect_hash(S, k_min=4, k_max=20, output="expression", solver="linprog", n_tries = 1500, n_prime=100)
print(expr)

import line_profiler
profiler = line_profiler.LineProfiler()
profiler.add_function(perfect_hash)
profiler.enable_by_count()
perfect_hash(S, k_min=3, k_max=20, solver="linprog", n_tries = 1500, n_prime=200)
profiler.print_stats(output_unit=1e-3)

from pbsig.perfect_hash import perfect_hash_dag

for i in range(100):
  x = np.random.choice(range(200), size=150, replace=False)
  f = perfect_hash_dag(x)
  assert all(np.array([f(xi) for xi in x]) == np.arange(len(x)))

n = 5500 # alphabet size
S = np.sort(np.random.choice(range(n), size=1350, replace=False)) # alphabet size
perfect_hash_dag(S, n_tries=1500, mult_max=3.0, n_prime=1000)

import line_profiler
profiler = line_profiler.LineProfiler()
profiler.add_function(perfect_hash_dag)
profiler.add_function(perfect_hash_dag)
profiler.enable_by_count()
perfect_hash_dag(S, n_tries=1500, mult_max=3.0, n_prime=1000)
profiler.print_stats(output_unit=1e-3)