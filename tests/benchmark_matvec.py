## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import *
from pbsig.utility import *
from pbsig.linalg import *

def generate_dataset(n: int = 15, d: int = 2):
  X = np.random.uniform(size=(n,d))
  K = delaunay_complex(X) 
  return X, K

## Generate random data set with weighted scalar product
X, S = generate_dataset(1000, 2)
fv = np.random.uniform(size=X.shape[0], low=0.0, high=1.0)

LM = up_laplacian(S, weight=lambda s: max(fv[s]), form='array')
LO = up_laplacian(S, weight=lambda s: max(fv[s]), form='lo')

## It is a bit faster! averaged over 30 runs! Maybe subclass!
import timeit
x = np.random.uniform(size=S.shape[0])
timeit.timeit(lambda: LO @ x, number=1500) # 0.057967034001194406
timeit.timeit(lambda: LM @ x, number=1500) # 0.014970820997405099
timeit.timeit(lambda: LO._matvec(x), number=1500) # 0.04564935999951558 

import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(lo._matvec)
profile.enable_by_count()
_ = eigsh(lo, k=5, which='LM', return_eigenvectors=False)
profile.print_stats(output_unit=1e-3)

eps, p = 0.1, 1
s1 = np.vectorize(lambda t: (t**p)/(t**p + eps**p))
s2 = np.vectorize(lambda t: (t**p)/(t**p + eps))
s3 = np.vectorize(lambda t: t/((t**p + eps**p)**(1/p)))
s4 = np.vectorize(lambda t: 1 - np.exp(-t/eps))

x = np.linspace(-1, 1, 1000)

for eps in [1e-12, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10]:
  plt.plot(x, s1(abs(x)), label=f"{eps:.3f}")
plt.gca().set_ylim(0, 1)
plt.legend()

plt.plot(x, s2(abs(x)))
plt.gca().set_ylim(0, 1)

for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10]:
  plt.plot(x, s3(abs(x)))
plt.gca().set_ylim(0, 1)


for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 10]:
  plt.plot(x, s4(abs(x)))
  plt.gca().set_ylim(0, 1)