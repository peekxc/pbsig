import numpy as np 
from pbsig.perfect_hash import perfect_hash

np.random.seed(1234)
n = 1500 # alphabet size
S = np.sort(np.random.choice(range(n), size=350, replace=False)) # alphabet size

g, expr = perfect_hash(S, k_min=4, k_max=4, output="expression", solver="linprog", n_tries = 1500, n_prime=100)
print(expr)

import line_profiler
profiler = line_profiler.LineProfiler()
profiler.add_function(perfect_hash)
profiler.enable_by_count()
perfect_hash(S, k_min=3, k_max=20, solver="linprog", n_tries = 1500, n_prime=200)
profiler.print_stats(output_unit=1e-3)
