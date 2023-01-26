import numpy as np
from scipy.spatial.distance import pdist, cdist
from itertools import * 
from pbsig.simplicial import * 
from pbsig.persistence import * 
from pbsig.vineyards import * 
import _persistence as pm

## Precompute center of each pixel + indices to index with 
def pixel_circle(n):
  vertex_pos = np.array(list(product(range(n), range(n)))) + 0.5
  center = np.array([[n/2,n/2]])
  x_dist = cdist(vertex_pos, center)/((n/2)*np.sqrt(2))

  ## Return a function
  def _circle(radius, width=0.15, e=-3.85):
    #I = np.zeros(shape=(n,n))
    return np.exp(e*(abs(x_dist - radius)/width)).reshape((n,n))
  return _circle

n = 9
C = pixel_circle(n)
X = C(0)
assert X.flags.c_contiguous

from pbsig.vis import plot_complex
from pbsig.simplicial import freudenthal, SimplicialComplex, MutableFiltration
S = freudenthal(C(0))
G = np.array(list(product(range(n), range(n))))
#normalize_unit = lambda x: (x - min(x))/(max(x) - min(x))
plot_complex(S, pos=G, color = C(0.56).flatten(), palette="gray", bin_kwargs = dict(lb=0.0, ub=1.0))


## Benchmark regular phcol 
from pbsig.vineyards import move_stats
OPS_PHCOL = [0]
pm.reduction_stats(True)
for p in np.linspace(0, 1, num=20):
  fv = C(p).flatten() # sorted by (r,c) 
  K = MutableFiltration(S, f=lambda s: max(fv[s]))
  D = boundary_matrix(K)
  V = eye(D.shape[0])
  I = np.arange(0, D.shape[1])
  R, V = pm.phcol(D, V, I)
  OPS_PHCOL.extend([pm.reduction_stats(False)])

## Benchmark vineyards
from pbsig.vineyards import vineyards_stats, transpose_rv
fv = C(0).flatten() # sorted by (r,c) 
K = MutableFiltration(S, f=lambda s: max(fv[s]))
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)
R = R.astype(int).tolil()
V = V.astype(int).tolil()
R = R.todense()
V = V.todense()
OPS_VINES = [vineyards_stats(reset=True)]
for p in np.linspace(0, 1, num=100):
  fv = C(p).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]), vines=True)
  OPS_VINES.append(vineyards_stats().copy())
  print(p)


## Benchmark moves 
fv = C(0).flatten() # sorted by (r,c) 
K = MutableFiltration(S, f=lambda s: max(fv[s]))
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)
R = R.astype(int).tolil()
V = V.astype(int).tolil()
R = R.todense()
V = V.todense()
from pbsig.vineyards import move_stats
OPS_MOVES = [move_stats(reset=True)]
for p in np.linspace(0, 1, num=20):
  fv = C(p).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]))
  #assert K.keys() do some test 
  OPS_MOVES.append(move_stats().copy())
  print(p)

## TODO: solve the bottlenecks to make experimentation feasible!
import line_profiler












## Onto the vineyards! 
from pbsig.persistence import *
from pbsig.vineyards import *

brightness = [np.ravel(C(x)) for x in np.linspace(0, 1, 100, endpoint=False)]

plt.plot(np.array(brightness)[0,:])
plt.plot(np.array(brightness)[1,:])
plt.plot(np.array(brightness)[:,24])

np.logical_and(fv0 == 1.0, fv1 == 1.0)

fv0, fv1 = np.ravel(C(0.0)), np.ravel(C(0.40))
K0 = MutableFiltration(S, f=lambda s: max(fv0[s]))
K1 = MutableFiltration(S, f=lambda s: max(fv1[s]))


schedule_tr, dom = linear_homotopy(dict(zip(K0.values(), K0.keys())), dict(zip(K1.values(), K1.keys())), plot=True, interval=(0,0.2891037310109982))


from scipy.sparse import identity

D0 = boundary_matrix(K0)
V0 = identity(D0.shape[1]).tolil()
R0 = D0.copy().tolil()
pHcol(R0, V0)
assert is_reduced(R0)
dgm0 = generate_dgm(K0, R0)

# lower_star_ph_dionysus(fv0, E, T)

## Execute naive vineyards
n_crit = 0
it = transpose_rv(R0, V0, schedule_tr)
F = list(K0.values())
for cc, (status,xval) in enumerate(zip(it, dom)):
  i = schedule_tr[cc]
  tau, sigma = F[i], F[i+1]
  assert is_reduced(R0)
  assert not(tau in sigma)
  print(f"{cc}: status={status}, nnz={R0.count_nonzero()}, n pivots={sum(low_entry(R0) != -1)}, pair: {(F[i], F[i+1])}")
  F[i], F[i+1] = F[i+1], F[i]
  fv = (1-xval)*fv0 + xval*fv1
  if status in [2,5,7]:
    n_crit += 1 
    dgm0_dn = lower_star_ph_dionysus(fv, E, T)
    dgm0_vi = generate_dgm(MutableFiltration(S, f=lambda s: max(fv[s])), R0)
    assert len(dgm0_vi) == (len(dgm0_dn[0]) + len(dgm0_dn[1]))


## Moves (global)
D = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
R = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
V = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
D = scipy.sparse.bmat([[None, D], [np.zeros((3,3)), None]]).tolil()
R = scipy.sparse.bmat([[None, R], [np.zeros((3,3)), None]]).tolil()
V = scipy.sparse.bmat([[np.eye(3), None], [None, V]]).tolil()
assert np.allclose(R.todense(), (D @ V).todense() % 2)
R, V = R.astype(int), V.astype(int)

## Move right (global)
#move_right(R, V, 3, 5)
#assert np.allclose(R.todense(), (permute_cylic_pure(D, 3, 5) @ V).todense() % 2)

## TODO: 
## 1. ensure face poset is respected (done; not needed!)
## 2. Implement greedy heuristic (done)

print(R)
print(V)
move_right(R, V, 0, 2)
print(R)
print(V)
np.allclose(R, permute_cylic_pure(D, i,j,"cols") @ V % 2 )
