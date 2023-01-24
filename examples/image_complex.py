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
move_right(R, V, 3, 5)
assert np.allclose(R.todense(), (permute_cylic_pure(D, 3, 5) @ V).todense() % 2)

## Move left (global)






## Moves (local)
D = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
R = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
V = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
assert np.allclose(R, (D @ V) % 2)

print(R)
print(V)
move_right(R, V, 0, 2)
print(R)
print(V)
np.allclose(R, permute_cylic_pure(D, i,j,"cols") @ V % 2 )


## Move left!
i,j = 0, 2

## First, obtain a coset representative that clears V[i:j,j]
add_column(V, j, V[:,[1]])
add_column(V, j, V[:,[0]])

add_column(R, j, R[:,[1]])
add_column(R, j, R[:,[0]])

## Then permute via move left 
permute_cylic(R, i, j, "cols", right=False)
permute_cylic(V, i, j, "both", right=False)

## Now reduce R through 
low_entry(R[:,i:(j+1)])

reduction_pHcol(D1, D2)

# piv = low_entry(R) 
# I = np.flip(np.arange(i,j+1)[V[i,i:(j+1)] != 0])
# dL, dR, dV = low_entry(R, I[0]), R[:,[I[0]]], V[:,[I[0]]]
# for k,l in pairwise(I):
#   tL, tR, tV = low_entry(R, l), R[:,[l]], V[:,[l]] # temporary 
#   add_column(R, k, tR)
#   add_column(V, k, tV)
#   if tL < dL: 
#     dL, dR, dV = tL, tR, tV

permute_cylic(R, i, j, "cols") ## change if full boundary matrix is used
permute_cylic(V, i, j, "both")

DS = lil_array(D)
cancel_column(DS, 1, DS[:,[0]])
cancel_column(D, 1, D[:,[0]])

i,j = 10, 25
V0[i,i:j]


# def move_right(R, V, i, j):



# import matplotlib.pyplot as plt
# plt.imshow(C(0.8))

# from pbsig.datasets import freudenthal_image


# freudenthal_image(C(0))


