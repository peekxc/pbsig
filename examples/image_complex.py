import numpy as np
from scipy.spatial.distance import pdist, cdist
from itertools import * 

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

n = 5
C = pixel_circle(5)
X = C(0)
assert X.flags.c_contiguous

## Freudenthal triangulation
from pbsig.utility import expand_triangles
nv = np.prod(X.shape)
V = np.fromiter(range(nv), dtype=int)
v_map = { (i,j) : v for v, (i,j) in enumerate(product(range(n), range(n))) } 

E = []
for (i,j) in product(range(n), range(n)):
  K13 = [(i-1, j), (i-1, j+1), (i, j+1)] # claw graph 
  E.extend([(v_map[(i,j)], v_map[e]) for e in K13 if e[0] in range(n) and e[1] in range(n)])
E = np.unique(np.array(E), axis=0)
T = expand_triangles(nv, E)

from pbsig.simplicial import SimplicialComplex, MutableFiltration
from itertools import chain

from pbsig.vis import plot_complex
S = SimplicialComplex(chain(V, E, T))
G = np.array(list(product(range(n), range(n))))
plot_complex(S, pos=G)


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



from pbsig.persistence import generate_dgm
dgm = generate_dgm(K, R0)

lower_star_ph_dionysus(f, E, T)
plot

from pbsig.vineyards import * 

# import matplotlib.pyplot as plt
# plt.imshow(C(0.8))

# from pbsig.datasets import freudenthal_image


# freudenthal_image(C(0))


