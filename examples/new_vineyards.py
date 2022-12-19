import numpy as np 
from scipy.sparse import *
from pbsig.simplicial import * 
from pbsig.persistence import * 
from pbsig.betti import * 
from pbsig.vineyards import * 

np.random.seed(1234)
X = np.random.uniform(size=(16,2))
S = delaunay_complex(X)
fv = X @ np.array([0,1])

## Build a lower-star filtration
K = MutableFiltration(S, f=lambda s: max(fv[s]))

## Do reduction 
D = boundary_matrix(K).tolil()
V = eye(D.shape[0]).tolil()
pHcol(D, V)

fl = X @ np.array([np.sqrt(2),np.sqrt(2)])
L = MutableFiltration(S, f=lambda s: max(fl[s]))

KK = { v:k for k,v in K.items() }
LL = { v:k for k,v in L.items() } 

tr = linear_homotopy(KK, LL)

from pbsig.vineyards import transpose_rv
h = transpose_rv(R, V, tr)

import time
from itertools import islice
for i in h:
  print(f"R shape: {R.shape}, nnz: {R.nnz}, Condition {i}")
  time.sleep(0.10)




