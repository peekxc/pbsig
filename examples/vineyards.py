import numpy as np
import scipy.sparse as sps

from pbsig.persistence import pHcol, boundary_matrix, reduction_pHcol, persistence_pairs

## Toy simplicial complex
V = [0,1,2,3,4,5,6,7]
E = [[0,1], [1,2], [0,2], [2,3], [3,4], [4,5], [5,6], [5,7], [6,7]]
T = [[0,1,2], [5,6,7]]
K = { 'vertices': V, 'edges' : E, 'triangles' : T }

## Boundary matrices
D1, D2 = boundary_matrix(K, p=1), boundary_matrix(K, p=2)

## Reduction
# pHcol(D1)
R1, R2, V1, V2 = reduction_pHcol(D1.tolil(), D2.tolil())
validate_decomp(D1, R1, V1, D2, R2, V2)

R0, V0 = sps.csc_matrix((0,len(V))), np.eye(len(V))
dgm0_index = persistence_pairs(R0, R1, collapse=False)
dgm1_index = persistence_pairs(R1, R2, collapse=False)


from pbsig.vineyards import permute_tr, transpose_dgm, linear_homotopy
R1, R2, V1, V2 = R1.tolil(), R2.tolil(), V1.tolil(), V2.tolil()
transpose_dgm(R1, V1, R2, V2, 0)


from pbsig import * 
from pbsig.simplicial import delaunay_complex
from pbsig.vineyards import line_intersection, inversion_dist, schedule, transpose_dgm
from operator import itemgetter
index = lambda i: itemgetter(i)
to_str = lambda X: [str(x) for x in X]

X = np.random.uniform(size=(16,2))
X = scale_diameter(X, 2.0)
X = X - (X.sum(axis=0)/X.shape[0])
K = delaunay_complex(X)

fig, ax = plot_mesh2D(X, K['edges'], K['triangles'], labels=True)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

## Choose filtration directions
F = np.hstack([f[:,np.newaxis] for (f, v) in rotate_S1(X, n=40)])
V,E,T = K.values()

res = []
fv0 = F[:,0]
fe0 = fv0[E].max(axis=1)
ft0 = fv0[T].max(axis=1)
vi, ei, ti = np.argsort(fv0), np.argsort(fe0), np.argsort(ft0)

Vf = dict(sorted(zip(to_str(V), fv0), key=index(1)))
Ef = dict(sorted(zip(to_str(E), fe0), key=index(1)))
Tf = dict(sorted(zip(to_str(T), ft0), key=index(1)))

D1, D2 = boundary_matrix(K, p=(1,2), f=((fv0,fe0), (fe0,ft0)))
D1, D2 = D1.tolil(), D2.tolil()
R1, R2, V1, V2 = reduction_pHcol(D1, D2)

## Handle deficient cases
R0, V0 = sps.lil_matrix((0,len(V))), sps.lil_matrix(np.eye(len(V)))
R3, V3 = sps.lil_matrix((len(T), 0)), sps.lil_matrix((0,0))
D0, D3 = R0, R3

assert validate_decomp(D0, R0, V0, D1, R1, V1)
assert validate_decomp(D1, R1, V1, D2, R2, V2)

v_names = np.array(list(Vf.keys()))
e_names = np.array(list(Ef.keys()))
P = persistence_pairs(R0, R1, f=(fv0[vi], fe0[ei]), names=(v_names, e_names))

P2 = barcodes(K, p=0, f=(np.sort(fv0), np.sort(fe0)))


for i in range(F.shape[1]):

  ## Extract current filtration values
  assert is_sorted(Vf.values()) and is_sorted(Ef.values())
  fv0 = np.fromiter(Vf.values(), dtype=float)
  fe0 = np.fromiter(Ef.values(), dtype=float)

  ## Update filtration values; maintain current key order
  Vf.update(zip(to_str(V), F[:,i]))
  Ef.update(zip(to_str(E), F[:,i][E].max(axis=1)))
  fv1 = np.fromiter(Vf.values(), dtype=float)
  fe1 = np.fromiter(Ef.values(), dtype=float)

  ## Schedule the transpositions
  ## f*1 should be ordered in the filtration order of f*0
  SV = schedule(fv0,fv1).min(axis=1)
  SE = schedule(fe0,fe1).min(axis=1)

  ## Sort the dictionary to reflect new order 
  Vf = dict(sorted(Vf.items(), key=index(1)))
  Ef = dict(sorted(Ef.items(), key=index(1)))

  # ST = schedule(ft0[ti],ft1[ti]).min(axis=1)
  # crit0 = [transpose_dgm(R0, V0, R1, V1, i) for i in SV]
  # crit1 = [transpose_dgm(R1, V1, R2, V2, i) for i in SE]
  
  ## Permute all the boundary matrices 
  for j in SV:
    transpose_dgm(R0, V0, R1, V1, j)
    permute_tr(D0, j, 'cols') 
    permute_tr(D1, j, 'rows')
    v_names[[j,j+1]] = v_names[[j+1,j]]
    # persistence_pairs(R0, R1, names=(v_names, e_names))
    assert validate_decomp(D0, R0, V0, D1, R1, V1)
  assert all(v_names == np.array(list(Vf.keys())))

  for j in SE:
    transpose_dgm(R1, V1, R2, V2, j)
    permute_tr(D1, j, 'cols') 
    permute_tr(D2, j, 'rows')
    e_names[[j,j+1]] = e_names[[j+1,j]]
    assert validate_decomp(D1, R1, V1, D2, R2, V2)
  assert all(e_names == np.array(list(Ef.keys())))

  # P = persistence_pairs(R0, R1, f=(np.sort(fv1), np.sort(fe1)), collapse=False, names=(v_names, e_names))
  # assert all((P['dgm'][:,1] - P['dgm'][:,0]) >= 0.0)

  # ## This should only pass after all the transpositions have finished
  # P = persistence_pairs(R0, R1, f=(np.sort(fv1), np.sort(fe1)), names=(v_names, e_names))

  # all(permute_tr(D1, i, 'cols') for i in SE)
  # all(permute_tr(D2, i, 'rows') for i in SE)
  # crit2 = [transpose_dgm(R2, V2, R3, V3, i) for i in ST]
  # D1 = boundary_matrix(K, p=1, f=(fv1,fe1))

  assert validate_decomp(D0, R0, V0, D1, R1, V1)
  assert validate_decomp(D1, R1, V1, D2, R2, V2)
  # validate_decomp()

  fv0 = np.fromiter(Vf.values(), dtype=float)
  fe0 = np.fromiter(Ef.values(), dtype=float)
  # dgm0 = persistence_pairs(R0, R1, f=(fv0, fe0), names=(v_names, e_names))
  # dgm1 = persistence_pairs(R1, R2, f=(np.sort(fe1), np.sort(ft1)))
  # dgm2 = persistence_pairs(R2, R3, f=(np.sort(ft1), np.array([])))


  # res.append(dgm0)
  print(i)
  break











np.argsort(fv0)
V[np.argsort(fv0)]
E[np.argsort(fe0),:]
plt.spy(D1)
plt.spy(R1)


plot_mesh2D(X, K['edges'], K['triangles'])
plot_dgm(dgm1)

SV = schedule(fv0[vi],fv1[vi]).min(axis=1)
SE = schedule(fe0[ei],fe1[ei]).min(axis=1)
ST = schedule(ft0[ti],ft1[ti]).min(axis=1)

crit0 = [transpose_dgm(R0, V0, R1, V1, i) for i in SV]
crit1 = [transpose_dgm(R1, V1, R2, V2, i) for i in SE]
crit2 = [transpose_dgm(R2, V2, R3, V3, i) for i in ST]

dgm0 = persistence_pairs(R0, R1, f=(np.sort(fv1), np.sort(fe1)))
dgm1 = persistence_pairs(R1, R2, f=(np.sort(fe1), np.sort(ft1)))
dgm2 = persistence_pairs(R2, R3, f=(np.sort(ft1), np.array([])))



# S, Sx = linear_homotopy(f0, f1, schedule=True) # transpositions + crossings
# S = schedule_transpositions(tr, np.argsort(f1[np.argsort(f0)]))



fe0 = f0[K['edges']].max(axis=1)
fe1 = f1[K['edges']].max(axis=1)



# Se, ex = linear_homotopy(fe0, fe1, schedule=False)
from itertools import combinations 
f0, f1 = fe0, fe1
P, Q = np.c_[np.repeat(0.0, len(f0)),f0], np.c_[np.repeat(1.0, len(f1)),f1]
n = P.shape[0]

results = []
for i,j in combinations(range(n), 2):
  pq = line_intersection((P[i,:], Q[i,:]), (P[j,:], Q[j,:]))
  if not(pq is None):
    results.append((i,j,pq[0],pq[1]))

results = np.array(results)
results = results[np.logical_and(results[:,2] >= 0.0, results[:,2] <= 1.0),:]


results = results[np.argsort(results[:,2]),:]
cross_points = results[:,[2,3]].astype(float)
transpositions = results[:,[0,1]].astype(int)


f0 = F[:,0][K['edges']].max(axis=1)
f1 = F[:,3][K['edges']].max(axis=1)

# f0, f1 = F[:,0], F[:,1]
from itertools import combinations 
from string import ascii_lowercase
m0 = { s:fs for s, fs in zip(ascii_lowercase[:len(f0)], f0) }
m1 = { s:fs for s, fs in zip(ascii_lowercase[:len(f1)], f1) }

m0_s = dict(sorted(m0.items(), key=lambda kv: (kv[1], kv[0])))
m1_s = dict(sorted(m1.items(), key=lambda kv: (kv[1], kv[0])))

a = list([ord(c)-97 for c in m0_s.keys()])
b = list([ord(c)-97 for c in m1_s.keys()])

n = len(f0)
P, Q = np.c_[np.repeat(0.0, len(f0)),f0], np.c_[np.repeat(1.0, len(f1)),f1]
results = []
for i,j in combinations(range(n), 2):
  i,j = a.index(i), b.index(j)
  pq = line_intersection((P[i,:], Q[i,:]), (P[j,:], Q[j,:]))
  if not(pq is None):
    results.append((i,j,pq[0],pq[1]))
results = np.array(results)
results = results[np.logical_and(results[:,2] >= 0.0, results[:,2] <= 1.0),:]
results = results[np.argsort(results[:,2]),:]


# inversion_dist(a,b)

ra0 = np.array([a.index(i) for i in results[:,0].astype(int)])
ra1 = np.array([a.index(i) for i in results[:,1].astype(int)])

w = sorted(zip(results[:,2], range(results.shape[0])), key=lambda r: (r[0], abs(ra1 - ra0)[r[1]]))
results2 = results[np.array([i[1] for i in w]),:]

p = np.fromiter(range(len(fe0)), int)
q = np.argsort(fe1)
inversion_dist(list(p),list(q))

  # ft = f[K['triangles']].max(axis=1)

tr, cr = linear_homotopy(f0, f1, schedule=False)


# a, b 
