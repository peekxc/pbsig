import numpy as np
import matplotlib.pyplot as plt
from pbsig.persistence import * 
from pbsig.apparent_pairs import * 


from pbsig.datasets import mpeg7

MPEG = mpeg7()


from pbsig.pht import pht_transformer, shape_center
preprocess = pht_transformer()
X = preprocess(MPEG[('turtle',1)])

from pbsig.utility import cycle_window
from pbsig.simplicial import SimplicialComplex, MutableFiltration
S = SimplicialComplex(cycle_window(np.arange(X.shape[0])))

fv = X @ np.array([1,0])
fv += abs(min(fv))

D = boundary_matrix(S)
from pbsig.linalg import up_laplacian
L = up_laplacian(S, weight = lambda s: max(fv[s]))
assert is_symmetric(L)
assert all(np.linalg.eigvalsh(L.todense()) >= -1e-14) # positive semi-definite check


from pbsig.linalg import numerical_rank
numerical_rank(L)



eigsh(L + (1e-12)*eye(L.shape[0]), which='LM', sigma = 0, return_eigenvectors=False)

import primme
primme.eigsh(L, k=1, which='LM')
primme.eigsh(L, k=L.shape[0]-1, ncv=500, which='LM', method="PRIMME_STEEPEST_DESCENT", return_eigenvectors=False, printLevel=5)
eigsh(L + (1e-6)*eye(L.shape[0]), which='SM', return_eigenvectors=False)
eigsh(L, k=1, which='SM', return_eigenvectors=False)
eigsh(L, k=1, sigma=0, which='LM', return_eigenvectors=False)
primme.eigsh(L, k=1, which='SM', maxiter=5000)
primme.eigsh(L+tol*eye(L.shape[0]), k=1, sigma=0, which='CGT', maxiter=5000, v0=np.c_[np.repeat(1, L.shape[0])])
primme.eigsh(L+tol*eye(L.shape[0]), k=1, sigma=0, which='LM', maxiter=5000, v0=np.c_[np.repeat(1, L.shape[0])])

primme.eigsh(L+tol*eye(L.shape[0]), k=1, tol=tol, which='LM', maxiter=15000, return_eigenvectors=False)
primme.eigsh(L+tol*eye(L.shape[0]), k=1, tol=tol, which='LM', maxiter=15000, method="PRIMME_STEEPEST_DESCENT", return_eigenvectors=False)
primme.eigsh(L+tol*eye(L.shape[0]), k=1, tol=tol, which='SM', maxiter=15000, method="PRIMME_STEEPEST_DESCENT", return_eigenvectors=False)
primme.eigsh(L+tol*eye(L.shape[0]), k=1, tol=tol, which='SA', maxiter=15000, method="PRIMME_GD", return_eigenvectors=False)


# from scipy.sparse.csgraph import structural_rank
# r = structural_rank(LS)


eigsh(LS, which='LM', return_eigenvectors=False)[0]
eigsh(LS + (1e-6)*eye(LS.shape[0]), which='SM', return_eigenvectors=False)

## Add a fraction of identity to the diagonal! Def!

L = L.tocsc()
max([abs(L[r,c] - L[c,r]) for r,c in zip(*L.nonzero())])

from pbsig.persistence import ph0_lower_star, lower_star_multiplicity
E = np.array(list(S.faces(1)))
dgm0 = ph0_lower_star(fv, E)

i,j = 0.0, 0.25
k,l = 0.28, 0.40


up_laplacian(S, w0=fv, w1=fv[E].max(axis=1), p=0)


X = np.random.uniform(size=(8, 2))
K = rips(X, p=2, diam=np.inf) # full complex (2-skeleton)

D1 = boundary_matrix(K, p=1)
D2 = boundary_matrix(K, p=2)

theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
X = np.c_[np.cos(theta), np.sin(theta)]
plot_rips(X, np.sort(pdist(X))[15])
plot_rips(X, np.sort(pdist(X))[25])
b, d = np.sort(pdist(X))[15], np.sort(pdist(X))[25]
PB = np.array([persistent_betti_rips(X, b = b, d = d, check_poset=True) for b, d in combinations(np.sort(pdist(X)), 2)])

plot_rips(X, diam=0.75)

## Check persistent betti is valid for all valid indices (i,j) respecting the face poset
from itertools import product, combinations
Edges = np.reshape(D1.indices, (D1.shape[1], 2))
Triangles = np.reshape(D2.indices, (D2.shape[1], 3))
for i,j in product(range(1, D1.shape[1]+1), range(1, D2.shape[1]+1)):
  all_faces = [list(combinations(tri, 2)) for tri in Triangles[:j,:]]
  all_faces = np.unique(np.reshape(np.array(all_faces).flatten(), (len(all_faces)*3, 2)), axis=0)
  E = rank_combs(Edges[:i,:], n=X.shape[0], k=2)
  F = rank_combs(all_faces, n=X.shape[0], k=2)
  pb = persistent_betti(D1, D2, i=i, j=j)
  if np.all(np.array([face in E for face in F])):
    assert pb >= 0
  

theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
x = np.c_[np.cos(theta), np.sin(theta)]
x += np.random.uniform(-0.01, 0.01, np.shape(x))
b,d = np.sort(pdist(x))[7] + 0.0001, np.sort(pdist(x))[18] - 0.0001
plot_rips(x, b)
plot_rips(x, d)
persistent_betti_rips(x, b, d, summands=True)


D1, ew = rips_boundary(x, p=1, diam=d, sorted=True)
D2, tw = rips_boundary(x, p=2, diam=d, sorted=True)

## Should match homology of K_d
Bd = (D1.shape[1] - np.linalg.matrix_rank(D1.A)) - np.linalg.matrix_rank(D2.A)

D1.data = np.sign(D1.data)*(np.maximum(b - np.repeat(ew, 2), 0.0))
D2.data = np.sign(D2.data)*(np.maximum(d - np.repeat(tw, 3), 0.0))

# U, E, Vt = np.linalg.svd(D1.A, full_matrices=False)
# X, S, Yt = np.linalg.svd(D2.A, full_matrices=False)
# PN = np.eye(D1.shape[1]) - Vt.T @ Vt
# PR = X @ X.T
# PI = 2*(PN @ np.linalg.pinv(PN + PR) @ PR)

# VN = Vt.T[:, np.flatnonzero(abs(E) <= 10*np.finfo(float).eps)]
# PN = VN @ VN.T
PN = np.eye(D1.shape[1]) - Vt.T @ Vt
PR = X @ X.T

# np.linalg.svd(PN @ D1.T)[1] is almost all 0, matching expectation
# np.linalg.svd(PR @ D2)[1] - S == 0 # true! 
PI = 2*(PN @ np.linalg.pinv(PN + PR) @ PR) ## PI basically all 0

## TODO: check 
np.linalg.svd(PI @ D1.T)[1]
np.linalg.svd(PI @ D2)[1]

## Von neumanns 
PI = np.linalg.matrix_power(PN @ PR, 15) # np.max(abs(PI)) still all 0! 

np.sum(abs(np.linalg.svd(np.c_[PI @ D1.T, PI @ D2])[1]))

## Ensure the spectral norm bound is always satisfied! 
assert np.sum(abs(np.linalg.svd(PI @ np.c_[D1.T.A, D2.A])[1])) <= np.max(abs(S))

persistent_betti_rips(x, b, d, summands=True)

# np.sum(abs(S))


# %% Begin circle example
import matplotlib.pyplot as plt
from persistence import *
from dms import circle_family
f, F = circle_family(58, sd=0.075)
b, d = 0.50, 0.75
plt.scatter(*f(1).T)

## b,d specific: find bounds on T
from scipy.optimize import root_scalar
lb = lambda t: np.max(pdist(f(t))) - b
ub = lambda t: np.min(pdist(f(t))) - d
max_t = root_scalar(ub, bracket=(0.01, 5.0)).root
min_t = root_scalar(lb, bracket=(0.01, 5.0)).root

## Start with the true objective
T = np.linspace(0, 1.0, 200) #np.linspace(0, max_t, 100)
PB = np.array([persistent_betti_rips(f(t), b, d) for t in T])
max_id = np.max(np.flatnonzero(PB == 1))

## Show complexes
fig, (ax1, ax2) = plt.subplots(1, 2) 
plot_rips(f(T[max_id]), diam=b, fig=fig, ax=ax1)
plot_rips(f(T[max_id]), diam=d, fig=fig, ax=ax2)

## Persistent Betti curve 
plt.plot(T, PB)
plt.xlabel("Time")
plt.ylabel("B_1^{b,d}")

## Plot relaxed objective
from autograd import grad
from betti import nuclear_constants, relaxed_pb, make_smooth_step, pb_gradient, smooth_grad

PB_D = np.array([persistent_betti_rips(f(t), b, d, summands=True) for t in T])
PB = PB_D[:,0] - (PB_D[:,1] + PB_D[:,2])

m1, m2 = nuclear_constants(F['n_points'], b, d, "global")
# m1, m2 = nuclear_constants(F['n_points'], b, d, "tight", f, T)
opts = dict(b=b, d=d, bp=b+0.25*(d-b), m1=m1, m2=m2, summands=False, p = 15, rank_normalize=True)
PBR_D = np.array([relaxed_pb(**(opts | dict(X=f(t), summands=True))) for t in T])
PBR = PBR_D[:,0] - (PBR_D[:,1] + PBR_D[:,2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), dpi=350)
ax1.plot(T, PB, c='black', label='f')
ax1.plot(T, PB_D[:,0], c='yellow', label='S')
ax1.plot(T, PB_D[:,1], c='orange', label='T1')
ax1.plot(T, PB_D[:,2], c='red', label='T2')
ax1.legend()
ax2.plot(T, PBR, c='blue', label='Relaxed f', zorder=30)
# ax2.plot(T, PBR*(PB+1), c='green', label='squared')
ax2.scatter(T, PBR, c='black', zorder=31, s=0.5)
ax2.plot(T, PBR_D[:,0], c='yellow', label='S')
ax2.plot(T, PBR_D[:,1], c='orange', label='T1')
ax2.plot(T, PBR_D[:,2], c='red', label='T2')
ax2.legend()

## NEED: instead of S(x), need | b - d |


## Very informative contiutuive plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3), dpi=350)
ax1.plot(T, PB_D[:,0], label='Betti S')
ax1.plot(T, PBR_D[:,0], label='Relaxed Betti S')
ax1.legend()
ax2.plot(T, PB_D[:,1], label='Betti T1')
ax2.plot(T, PBR_D[:,1], label='Relaxed Betti T1')
ax2.legend()
ax3.plot(T, PB_D[:,2], label='Betti T2')
ax3.plot(T, PBR_D[:,2], label='Relaxed Betti T2')
ax3.legend()

## Approximation error for each relaxation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3), dpi=350)
ax1.plot(T, np.abs(PB_D[:,0] - PBR_D[:,0]), label='Betti S Error')
ax2.plot(T, np.abs(PB_D[:,1] - PBR_D[:,1]), label='Betti T1 Error')
ax3.plot(T, np.abs(PB_D[:,2] - PBR_D[:,2]), label='Betti T2 Error')



## Interpolating f(x) = x -> f(x) = 1
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
knot = lambda alpha: (1-alpha)*np.array([0.5, 0.5]) + alpha*np.array([0.0, 1.0])
from scipy.interpolate import CubicSpline
# rank_interpolant = lambda alpha: CubicSpline([0.0, knot(alpha)[0], 1.0], [0.0, knot(alpha)[1], 1.0], bc_type="natural")

# %% Benchmark 
from line_profiler import LineProfiler
profiler = LineProfiler()
profiler.add_function(persistent_betti_rips)
profiler.enable_by_count()
PB = persistent_betti_rips(f(np.median(T)), b, d, summands=True)
profiler.print_stats(output_unit=1e-3)

# m1, m2 = nuclear_constants(p['n_points'], b, d, "global")
profiler = LineProfiler()
profiler.add_function(relaxed_pb)
profiler.enable_by_count()
relaxed_pb(X=f(np.median(T)), b=b, d=d, bp=b+0.25*(d-b), m1=m1, m2=m2, summands=True, p = 15, rank_normalize=True)
profiler.print_stats(output_unit=1e-3)

from scipy.linalg.interpolative import id_to_svd, interp_decomp
from scipy.linalg.interpolative import svd as id_svd
k = 5
A = D1[:,:10].A
idx, proj = interp_decomp(A, eps_or_k=k)
B = A[:,idx[:k]]
P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
id_to_svd(B, idx, proj)

## B spline 
from scipy import interpolate
tl = (1-0.80)*np.array([0.5, 0.5]) + 0.80*np.array([0.0, 1.0])
x = np.array([0.0, tl[0], 1.0])
y = np.array([0.0, tl[1], 1.0])
t, c, k = interpolate.splrep(x, y, s=2, k=1) # s = smoothness
spline = interpolate.BSpline(t, c, k, extrapolate=False)
plt.plot(x, y, 'bo', label='Original points')
plt.plot(np.linspace(0, 1, 100), spline(np.linspace(0, 1, 100)), 'r', label='BSpline')


# %%  Bezier curve solution (the solution!)
beta = 1.0
tl = (1-beta)*np.array([0.5, 0.5]) + beta*np.array([0.0, 1.0])
x = np.array([0.0, tl[0], 1.0])
y = np.array([0.0, tl[1], 1.0])
p0, p1, p2 = np.array([x[0], y[0]]), np.array([x[1], y[1]]), np.array([x[2], y[2]])

P = np.exp(np.linspace(np.log(1), np.log(6), 20))
for p in P:
  f = lambda t: p1 + (1-t)**p * (p0 - p1) + t**p * (p2 - p1)
  dom = np.linspace(0, 1, 100)
  rng = np.array([f(x) for x in dom])
  plt.plot(*rng.T)

# %% Relaxation of smooth step
beta = 1.0
tl = (1-beta)*np.array([0.5, 0.5]) + beta*np.array([0.0, 0.0])
x = np.array([0.0, tl[0], 1.0])
y = np.array([1.0, tl[1], 0.0])
p0, p1, p2 = np.array([x[0], y[0]]), np.array([x[1], y[1]]), np.array([x[2], y[2]])
P = np.exp(np.linspace(np.log(1), np.log(6), 20))
for p in P:
  f = lambda t: p1 + (1-t)**p * (p0 - p1) + t**p * (p2 - p1)
  dom = np.linspace(0, 1, 100)
  rng = np.array([f(x) for x in dom])
  plt.plot(*rng.T)

# %% 
def rank_interpolant(alpha: float):
  p0, p3 = np.array([0, 0]), np.array([1, 1])
  #p1 = knot(alpha)
  p1 = (1-alpha)*p0 + alpha*np.array([0, 1])
  p2 = (1-alpha)*p3 + alpha*np.array([0, 1])
  def f(t):
    #P = np.array([p1, (1-t)**2 * (p0 - p1), t**2 * (p2 - p1)])
    fx = ((1-t)**3 * p0) + (3*(1-t)**2 * t * p1) + (3 * (1-t) * t**2 * p2) + t**3 * p3
    return(fx)
  return(f)

for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
  f = rank_interpolant(alpha)
  x = np.linspace(0, 1, 100, endpoint=True)
  # knots = np.c_[[0.0, knot(alpha)[0], 1.0], [0.0, knot(alpha)[1], 1.0]]
  ax.plot(*np.array([f(xi) for xi in x]).T)
  p0, p3 = np.array([0, 0]), np.array([1, 1])
  p1 = (1-alpha)*p0 + alpha*np.array([0, 1])
  p2 = (1-alpha)*p3 + alpha*np.array([0, 1])
  ax.scatter(*np.c_[p0,p1,p2,p3])

## Can achieve the result using n-degree Bezier curves w/ control points [p0, ..., pn]
## where pi = knot(alpha) for some alpha \in [0, 1], 0 < i < n.
#knot = lambda alpha: (1-alpha)*np.array([0.5, 0.5]) + alpha*np.array([0.0, 1.0])
import bezier
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(0, 1.025)
ax.set_ylim(0, 1.025)

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
  knot_pt = knot(alpha)
  for n in range(0, 8):
    cp = np.reshape(np.tile(np.array(knot_pt), n), (n, 2))
    cp = np.vstack(([0,0], cp, [1,1]))
    nodes = np.asfortranarray(cp.T)
    curve = bezier.Curve(nodes, degree=nodes.shape[1]-1)
    curve.plot(1000,ax=ax)

# other datasets
from dms import moving_circles
f, P = moving_circles(3, 100)

from tallem.dimred import rnn_graph
from scipy.sparse.csgraph import floyd_warshall
G = rnn_graph(f(0), p=0.15)
D = floyd_warshall(G)
R = rips(pdist(f(0)), diam=d, p=2)

b, d = 0.30, 0.35
PB_D = np.array([persistent_betti_rips(f(t), b, d, summands=True) for t in T])
PB = PB_D[:,0] - (PB_D[:,1] + PB_D[:,2])
# plt.plot(T, PB)

m1, m2 = nuclear_constants(P['n_points'], b, d, "tight", f, T)
opts = dict(b=b, d=d, bp=b+0.25*(d-b), m1=m1, m2=m2, summands=False, p = 15, rank_normalize=True)
PBR_D = np.array([relaxed_pb(**(opts | dict(X=f(t), summands=True))) for t in T])
PBR = PBR_D[:,0] - (PBR_D[:,1] + PBR_D[:,2])

plt.plot(T, PB)
plt.plot(T, PBR)

plot_rips(f(t), diam=b)
plot_rips(f(t), diam=d)

## INcorrect 
t = T[int(np.median(np.flatnonzero(~(PBR_D[:,2] <= PB_D[:,2]))))]
s1,s2,s3 = persistent_betti_rips(f(t), b, d, summands=True)
t1,t2,t3 = relaxed_pb(f(t), b=b, d=d, bp=b+0.10*(d-b), m1=m1, m2=m2, summands=True, p = 1)

bump = lambda x: np.exp(-1/(1-x**2))
plt.plot(np.linspace(0, 1, 100), bump(np.linspace(0, 1, 100))**2)

e = 4
domain = np.linspace(0, e, 100)
plt.plot(domain, e-domain)
plt.plot(domain, ((e-domain)/e)**1.0*e)
plt.gca().set_aspect('equal')

## Gradients 
S = make_smooth_step(opts['b'], opts['bp'])
SG = smooth_grad(opts['b'], opts['bp'])
# SG = grad(S)
GS = np.array([pb_gradient(f, t, b, d, SG, summands=True) for t in T])

## Plot objective + relaxation + gradient sign without normalizing using m1, m2
G = -GS[:,0] - (GS[:,1] + GS[:,2])
plt.plot(T, PBR, c='blue', label='Relaxed f')
plt.plot(T, PB, c='black', label='f')
g_col = np.array(['green', 'gray', 'red'])[np.array(np.sign(G)+1, dtype=int)]
plt.scatter(T, PBR, c=g_col, s=15.5)

## With m1, m2 normalization
G = -GS[:,0] - (1/m1)*GS[:,1] - (1/m2)*GS[:,2]
plt.plot(T, PBR, c='blue', label='Relaxed f')
plt.plot(T, PB, c='black', label='f')
g_col = np.array(['green', 'gray', 'red'])[np.array(np.sign(G)+1, dtype=int)]
plt.scatter(T, PBR, c=g_col, s=15.5)

## Zoom to specific area
fig = plt.figure(dpi=220, figsize=(8,8))
ax = fig.gca()
min_id = np.min(np.flatnonzero(PB == 1))-25
max_id = np.max(np.flatnonzero(PB == 1))+25
GM = -GS[min_id:max_id,0] - (1/m1)*GS[min_id:max_id,1] - (1/m2)*GS[min_id:max_id,2]
ax.plot(T[min_id:max_id], PBR[min_id:max_id], c='blue', label='Relaxed f')
ax.plot(T[min_id:max_id], PB[min_id:max_id], c='black', label='f')
g_col = np.array(['green', 'gray', 'red'])[np.array(np.sign(GM)+1, dtype=int)]
ax.scatter(T[min_id:max_id], PBR[min_id:max_id], c=g_col, s=15.5)
plt.show()


# from autograd import grad
# g = grad(S)
# grad_S = np.array([g(x).item() for x in np.linspace(b, bp, 100)])
# Sg = lambda x: 1.5*((x-b)/(bp-b))**2 - 3*((x-b)/(bp-b))
# g(b + 0.05)
# Sg(b + 0.05)
# h = 100*np.finfo(float).eps
# (S(b+0.05+h)-S(b+0.05-h))/2*h


T = np.linspace(0, max_t, 100)

#PBR = np.array([relaxed_pb(f(t), b, d, b + (d-b)*0.10, m1, m2) for t in T])
PB = np.array([persistent_betti_rips(f(t), b, d) for t in T])
PBR = np.array([relaxed_pb(f(t), b, d, b + (d-b)*0.10, m1, m2, summands=True) for t in T])
plt.plot(T, PBR[:,0], c='yellow', label='S')
plt.plot(T, PBR[:,1], c='orange', label='T1')
plt.plot(T, PBR[:,2], c='red', label='T2')
plt.plot(T, PBR[:,0]-(PBR[:,1]+PBR[:,2]), c='blue', label='Relaxed f')
plt.plot(T, PB, c='black', label='f')
plt.gca().legend()

S = make_smooth_step(b, b+1.0*(d-b))

pb_gradient(f, t, b, d, S, summands=True)

G = np.array([pb_gradient(f, t, b, d, S, summands=False) for t in T])
G = np.array([pb_gradient(f, t, b, d, S, summands=True) for t in T])
G = G[:,0] - ((1/m1)*G[:,1] + (1/m2)*G[:,2])
plt.plot(T, G, c='green', label='gradient')
plt.plot(T, PB, c='black', label='f')
plt.plot(T, PBR[:,0]-(PBR[:,1]+PBR[:,2]), c='blue', label='Relaxed f')

#plt.scatter(T, PBR, c='red', s=0.05)
# %%

for t in T: 
  dx = pdist(f(t))
  plt.scatter(np.repeat(t, len(dx)), dx)
plt.axhline(y=b, color='b', label='b')
plt.axhline(y=d, color='red', label='d')
# for i in range(1000):
#   M = np.random.uniform(size=(10,10))
#   assert np.max(np.linalg.svd(M)[1]) <= np.sqrt(np.max(np.sum(abs(M), axis=0)) * np.max(np.sum(abs(M), axis=1)))





D1.data = np.sign(D1.data)*(np.maximum(b - np.repeat(ew, 2), 0.0))
D2.data = np.sign(D2.data)*(np.maximum(d - np.repeat(tw, 3), 0.0))
  

import cppimport.import_hook
import boundary
import numpy as np
from scipy.spatial.distance import pdist
from scipy.sparse import csc_matrix

X = np.random.uniform(size=(100, 2))
D = pdist(X)

import timeit

def build_D1():
  cdata, cindices, cindptr = boundary.rips_boundary_matrix_1(D, X.shape[0], 0.50)
  D1 = csc_matrix((cdata, cindices, cindptr), shape=(X.shape[0], len(cindptr)-1))
  return(D1)

build_D1_orig = lambda: rips_boundary(D, p=1, diam=0.50)
timeit.timeit(build_D1_orig, number=30)/timeit.timeit(build_D1, number=30) # 450.85x speedup 
timeit.timeit(build_D1, number=30)

cdata, cindices, cindptr, tw = boundary.rips_boundary_matrix_2_dense(D, X.shape[0], 0.50)
mask = np.cumsum(D <= 0.50)-1
cindices = mask[cindices]  
D2 = csc_matrix((cdata, cindices, cindptr), shape=(np.sum(D <= 0.50), len(cindptr)-1))


from scipy.spatial import Delaunay
Delaunay(X).simplices.shape

import scipy
scipy.special.binom(145, 3)

from persistence import rips_boundary
D1, (vw, ew) = rips_boundary(D, p=1, diam=0.12)
D2, (ew, tw) = rips_boundary(D, p=2, diam=0.12)

from apparent_pairs import *
n = X.shape[0]
TW = np.array([np.max(D[[rank_C2(i,j,n), rank_C2(i,k,n), rank_C2(j,k,n)]]) for (i,j,k) in combinations(range(X.shape[0]), 3)])
np.sum(TW <= 0.50)

np.sort(cdata[cdata > 0])
np.sort(TW[TW <= 0.50])

np.sort(np.unique(np.abs(cdata)))

from scipy.sparse import csc_matrix
A = np.random.uniform(size=(5,5))
A[A <= 0.60] = 0.0 
A = csc_matrix(A)
A.eliminate_zeros()

A.A

# %%
