import numpy as np
import matplotlib.pyplot as plt
from persistence import * 
from apparent_pairs import * 

X = np.random.uniform(size=(16, 2))
K = rips(X, p=2, diam=np.inf) # full complex (2-skeleton)


D1, ew = rips_boundary(X, p=1, diam=0.5, sorted=True)
D2, tw = rips_boundary(X, p=2, diam=0.5, sorted=True)
R1, R2, V1, V2 = reduction_pHcol(D1, D2, clearing=True)


R2_lows = low_entry(R2)
r_nz, c_nz = R2_lows[R2_lows > -1], np.flatnonzero(R2_lows > -1)

from itertools import product
for b,d in product(r_nz, c_nz):
  # row_indices = R2_lows[np.bitwise_and(R2_lows <= b, R2_lows != -1)]
  K = np.flatnonzero(np.bitwise_and(R2_lows <= b, R2_lows != -1)[:(d+1)])
  if len(K) == 0: 
    continue
  R2_tmp = R2[:, K].A
  D2_tmp = D2[:, K].A
  V2_tmp = V2[np.ix_(K, K)].A
  if np.linalg.matrix_rank(R2_tmp) != np.linalg.matrix_rank(D2_tmp):
    print(f"Rank failure | b: {b}, d: {d}")
    # print(f"R2: {np.linalg.matrix_rank(R2_tmp)}, D2: {np.linalg.matrix_rank(D2_tmp)}")

  W = R2_tmp @ np.linalg.inv(V2_tmp)
  W[W != 0] = 1
  # np.sum(W, axis=0) != np.sum(abs(D2_tmp), axis=0) 
  if not(np.all((R2_tmp @ np.linalg.inv(V2_tmp)) == D2_tmp)):
    print(f"Boundary failure | b: {b}, d: {d}")
