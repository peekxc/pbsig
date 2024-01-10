# %% Imports
from typing import *
import numpy as np
import splex as sx
import splex as sx
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from pbsig.betti import SpectralRankInvariant
from pbsig.datasets import noisy_circle
from pbsig.interpolate import ParameterizedFilter
from pbsig.persistence import ph
from pbsig.rivet import bigraded_betti, push_map
from pbsig.vis import *
from scipy.interpolate import BSpline, splrep
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from pbsig.datasets import PatchData
from pbsig.vis import figure_complex, show

output_notebook()

# %% Get the data, show the basis vectors  
Klein = PatchData(lazy=False)
show(Klein.figure_basis())

# %% Show the projection onto first 2 eigenvectors 
p = figure(width=300, height=300)
p.scatter(*Klein.project_2d().T, color = "#80808010", size=2)
show(p)

# %% Show some landmarks
p = Klein.figure_patches(figure=p, size=0.25)
show(p)

# %% Show where the closest data to the basis vectors lie
# X = klein_data.project_2d()
# min_indices = [np.argmin(np.linalg.norm(np.abs(klein_data.patch_data - basis), axis=1)) for basis in klein_data.dct_basis.T]
# p.scatter(*X[min_indices,:].T, color='blue', size=3.5)
# p.scatter(*X[min_indices[:2],:].T, color='red', size=3.5)
# show(p)

# %% Alternative: use data shader 
p = Klein.figure_shade()
p = Klein.figure_patches(figure=p, size=0.25)
show(p)

# %% Stratified sampling 
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma
from pbsig.csgraph import neighborhood_graph
from scipy.sparse.csgraph import floyd_warshall
from pbsig.shape import landmarks, stratify_sample
from more_itertools import collapse

## Get a KNN density estimate
k = 30
Klein.basis = "dct"
patch_kdtree = KDTree(Klein.patch_data)
knn_dist, knn_ind = patch_kdtree.query(Klein.patch_data, k=k)
vol_ball = lambda d: np.pi**(d/2) * np.reciprocal(gamma(d/2 + 1))
d = Klein.patch_data.shape[1]
knn_density = ((k - 1) / d) * np.reciprocal(vol_ball(8) * knn_dist.max(axis=1))

# stratified_ind = stratify_sample(50, knn_density)
# density_threshold = np.quantile(knn_density, 0.95)
# knn_density[knn_density >= density_threshold] = density_threshold

np.random.seed(1234)
strata_weights = np.ones(25)
# strata_weights[-5:] = 0
_, edges = np.histogram(knn_density, 25)
between = lambda x, a, b: x[np.logical_and(a <= x, x <= b)]
interval_weight = np.array([between(knn_density, a, b).sum() for a,b in pairwise(edges)])
strata_weights = interval_weight / np.sum(interval_weight)

knn_strata = stratify_sample(250, knn_density, return_strata = True, n_strata = len(strata_weights), weights = strata_weights)
strata = []
for k, v in knn_strata.items():
  s_ind = v['stratum']
  if v['n'] > 0: 
    l_ind, _ = landmarks(Klein.patch_data[s_ind,:], k=v['n'])
    strata.append(s_ind[l_ind])
stratified_ind = np.array(list(collapse(strata)))

## alternative approach
# stratified_ind = np.random.choice(np.arange(len(knn_density)), size = 500, replace=False, p=knn_density / np.sum(knn_density))
# quantiles = np.quantile(knn_density, q=np.linspace(0, 1, 10)[1:])
# np.bincount(np.searchsorted(quantiles, knn_density))
# %% Show the stratified sampling
X = Klein.project_2d()
p = figure(width=300, height=300)
p = figure_scatter(X, fill_color = "#808080", fill_alpha = 0.15, size=2, line_width=0.0, figure = p)
p = figure_scatter(X[stratified_ind], fill_color = "blue", fill_alpha = 0.55, size=3, line_width=0.0, figure = p)
show(p)

# %% Compute the geodesic distances on the KNN graph
patch_knn_graph = neighborhood_graph(Klein.patch_data, k = k, weighted=True).tocsr()
# patch_geodesic = squareform(floyd_warshall(patch_knn_graph.tocsr(), directed=False))
patch_knn_graph.indptr = patch_knn_graph.indptr.astype(np.int32)
patch_knn_graph.indices = patch_knn_graph.indices.astype(np.int32)
patch_sps = shortest_path(patch_knn_graph, indices=stratified_ind)
patch_geodesic = patch_sps[:,stratified_ind]
patch_geodesic = squareform((patch_geodesic + patch_geodesic.T)/2)
# shortest_path(patch_knn_graph, directed=False, indices = 0)

#%% Form the two filter functions to create the bifiltration of
normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
n_codensity = normalize(max(knn_density[stratified_ind]) - knn_density[stratified_ind])
n_geodesic = normalize(patch_geodesic)
codensity_filter = sx.lower_star_filter(n_codensity)
geodesic_filter = sx.flag_filter(n_geodesic)

# %% Form the complex + the two filter functions
from scipy.sparse.csgraph import minimum_spanning_tree
print(f"Is connected? {not(np.any(np.isinf(patch_geodesic)))}")
f1 = codensity_filter
f2 = geodesic_filter
connected_diam = np.max(minimum_spanning_tree(squareform(n_geodesic)).data)
S = rips_complex(n_geodesic, radius=(connected_diam * 1.35) / 2, p = 2)

# %% Try to reduce the size of the complex w/ quadric decimation
# from pbsig.simplicial import simplify_complex
# from pbsig.linalg import pca, cmds 
# # cmds(patch_geodesic, 3)
# landmark_xyz = pca(Klein.patch_data, 3)[stratified_ind,:]
# # S_new, xyz = simplify_complex(S, landmark_xyz)

# from fast_simplification import simplify
# triangles = np.array(sx.faces(S, 2))
# pos = landmark_xyz.astype(np.float32) 
# pos_new, tri_new = simplify(pos, triangles, target_reduction = 0.50, agg = 1)
# simplicial_complex(tri_new, form="tree")

# simplify(Klein.patch_data[:,[0,1,2]].astype(np.float32), triangles, target_count = 100, agg = 9)[1].shape

# pc = Klein.patch_data[:,[0,1,2]].astype(np.float32)

# import pyfqmr
# mesh_simplifier = pyfqmr.Simplify()
# mesh_simplifier.setMesh(landmark_xyz, triangles)
# mesh_simplifier.simplify_mesh(target_count = 50000, aggressiveness=7, preserve_border=True, verbose=10)



# %% Sparsify with GUDHI 
import gudhi
sparse_S = gudhi.RipsComplex(distance_matrix=squareform(n_geodesic), max_edge_length=(connected_diam * 1.35), sparse=0.85)
sparse_st = sparse_S.create_simplex_tree(max_dimension=2)
# sparse_st.num_vertices()
print(f"Number simplices: {sparse_st.num_simplices()}")
S = sx.simplicial_complex([s for s, fs in sparse_st.get_simplices()], form='tree')
sf_map = { sx.Simplex(s) : fs for s,fs in sparse_st.get_simplices() }
s_values = np.array([sf_map[s] for s in S])
geodesic_filter = sx.fixed_filter(S, s_values)
f2 = geodesic_filter

# K = sx.RankFiltration([(k,s) for s,k in sparse_st.get_filtration()])
# sparse_st.compute_persistence()
# sparse_st.persistence_pairs()
# S_sparse = sx.simplicial_complex([s for s, v in list(sparse_st.get_simplices())], form='rank')

# %% Compute the hilbert function 
from pbsig.rivet import figure_betti, anchors, push_map
BI = bigraded_betti(S, f1, f2, p=1, xbin=30, ybin=30)

# %% Show the hilbert function
p = figure_betti(BI, width=350, height=350, highlight=5, size=5) #y_range = (0.95*connected_diam, connected_diam*1.60)))
show(p)

# %% Choose the intercept + angle by hand 
offset, angle = 0.11, 29
line_f = lambda x, theta: np.tan(np.deg2rad(theta)) * x + offset
p = figure_betti(BI, width=350, height=350, show_coords=True, highlight=5, size=5) #y_range = (0.95*connected_diam, connected_diam*1.60)))
p.line(x=[0.0, 1.0], y=[line_f(0.0, angle), line_f(1.0, angle)], color='red')

angle1, angle2 = 20, 39
p.line(x=[0.0, 1.0], y=[line_f(0.0, angle1), line_f(1.0, angle1)], color='blue')
p.line(x=[0.0, 1.0], y=[line_f(0.0, angle2), line_f(1.0, angle2)], color='blue')
show(p)

# %% Apply the push map to get a single filtration; look at its diagram
Y = np.c_[f1(S), f2(S)]
YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)

def push_fast(X, a, b):
  x, y = X[:,0], X[:,1]
  cond = (x * a + b) < y
  tmp_0 = np.where(cond, (y - b) / a,  x)
  tmp_1 = np.where(cond, y, a * x + b)
  return np.sqrt(tmp_0 ** 2 + (b - tmp_1) ** 2)

# timeit.timeit(lambda: push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset), number = 5) 
# # 1.5 s
# timeit.timeit(lambda: push_fast(Y, a=np.tan(np.deg2rad(angle)), b=offset), number = 5)
# 0.002796350046992302 seconds 

## This should show 5 points in H1 with high persistence
from pbsig.persistence import ph 
from pbsig.vis import figure_dgm
filter_angle = fixed_filter(S, YL[:,2])

# pairs = apparent_pairs(S_sparse, filter_angle, p=1)

## Raw persistence
K = sx.RankFiltration(S, filter_angle)
dgms = ph(K)
show(figure_dgm(dgms[1]))

a,b = 0.60, 0.70

#%% First, verify we get 5 in the spectral rank invariant
from pbsig.betti import BettiQuery, betti_query
# t0,t1,t2,t3 = betti_query(S, filter_angle, p = 1, i=a, j=b, matrix_func=lambda x: x, terms=True)
bq = BettiQuery(S, f=filter_angle, p=1)
bq.sign_width = 0.0

## Try to create an objective function 
from primate.trace import hutch

bq.q_solver = lambda L: numrank(L, atol = 0.50, confidence=0.90)  # hutch(L, maxiter=5)
bq.p_solver = lambda L: numrank(L, atol = 0.50, confidence=0.90)
t0, t1, t2, t3 = list(bq.generate(i=a, j=b, mf=lambda x: x))[0]

print(f"{np.sum(t0)}, {t1}, {t2}, {t3}")
np.sum(t0) - t1 - t2 + t3

tikhonov = lambda x: x / (x + 1e-2)
mf = lambda L: hutch(L, fun=tikhonov, rtol=0.10, maxiter=300)
bq.q_solver = mf
bq.p_solver = mf
t0, t1, t2, t3 = list(bq.generate(i=a, j=b, mf=np.sum))[0]

print(f"{np.sum(t0)}, {t1}, {t2}, {t3}")
np.sum(t0) - t1 - t2 + t3

## 
from pbsig.interpolate import ParameterizedFilter

## What is a parametrized filter? It should 
# ParameterizedFilter(
#   S, 
#   family=[push_fast(Y, a=np.tan(np.deg2rad(theta)), b=offset) for theta ], 
# )

for theta in np.linspace(angle1, angle2, 20):
  push_f = push_fast(Y, a=np.tan(np.deg2rad(theta)), b=offset)
  bq(i=a, j=b, mf=np.sum)


numrank(bq.Lq.operator(), atol = 5.0, confidence=0.99)


from scipy.sparse.linalg import eigsh
ew = eigsh(bq.Lp.operator(), k=248)[0]
np.sum(ew >= 1e-2)
numrank(bq.Lp.operator(), atol=0.0, plot=True, maxiter=300)
np.sum(np.abs(bq.Lp.operator().todense()).sum(axis=1) > 0)
A = bq.Lp.operator()
v = np.random.choice([-1.0, +1.0], size=A.shape[0]) / np.sqrt(249)
np.sum(A @ v != 0)

B = A.todense()[:10,:10]
B[1,1] = 1

np.unique(A.row)

rc = np.unique(np.append(A.row, A.col))
rc_map = { ii : i for i,ii in enumerate(rc) }
I = np.array([rc_map[i] for i in A.row])
J = np.array([rc_map[j] for j in A.col])
B = coo_array((A.data, (I,J)), shape=(len(rc), len(rc)))
numrank(B)

A = A.tocsr()
A.indices
A.indptr





numrank(B, verbose=True)




# 177.0 - 59 - 322 + 209 == 5! 

# t0,t1,t2,t3 = list(bq.generate(i=a, j=b, mf=lambda x: x))[0] # gave up after 12minutes


## Direct method
A_dense = bq.Lq.operator().todense()
np.linalg.matrix_rank(A_dense) # 8966, out of 9215 -- takes 4m 12s

## Implicit Lanzos via ARPACK 
from scipy.sparse.linalg import eigsh
eigsh(bq.Lq.operator(), k = bq.Lq.shape[1]-1, tol = 1e-6, maxiter=10e5, return_eigenvectors=False)
## didn't finish even after 21 minutes

# numrank(bq.Lq.operator(), atol=5, verbose=True, quad="fttr", plot=True)
# from primate.trace import hutch
# hutch(bq.Lq.operator(), fun="numrank", threshold=1e-2, atol = 5, plot=True, verbose=True, num_threads=1)

# eigsh(bq.Lq.operator(), k = 10, tol = 1e-6, maxiter=10e5, which='SM', return_eigenvectors=False)

import timeit
x = np.random.uniform(size=bq.Lq.shape[0])
timeit.timeit(lambda: bq.Lq.operator() @ x, number = 1500)
timeit.timeit(lambda: D @ (D.T @ x), number = 1500)

## This is the fastest, but not by much (~10%)
# D = bq.Lq.bm.tocsc()
# timeit.timeit(lambda:  D @ (D.T @ x), number = 1500)

## Memory reduction is only about 57% 
# (D.indices.nbytes + D.indptr.nbytes + D.data.nbytes) // 1024
# L = bq.Lq.operator()
# (L.row.nbytes + L.col.nbytes + L.data.nbytes) // 1024

## SLQ 
from primate.functional import numrank
# bq.Lq.reweight(np.where(bq.sw <= 0.70, 1.0, 0.0), np.where(bq.fw > 0.55, 1.0, 0.0))
bq.Lq.reweight(np.where(bq.sw <= 0.70, 1.0, 0.0), np.ones(len(bq.fw)))
A = bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T

# from primate.plotting import figure_trace
# wut = sl_trace(bq.Lq.operator(), fun = "numrank", maxiter=2000, orth = 10, deg = 50, num_threads=8)
# show(figure_trace(wut))
# show(figure_trace(wut[wut < 9100]))
# # np.min(wut)
# # np.max(wut)
# np.mean(wut)
# np.mean(wut[wut < 9100])


nw = sl_gauss(bq.Lq.operator(), n=1500)

## Rank is likely ~ 71 
n = bq.Lq.operator().shape[0]
tol = nw[:,0].max() * np.finfo(np.float32).eps
np.sum(np.where(nw[:,0] > tol, 1.0, 0) * nw[:,1]) * n / nw.shape[0]

from scipy.sparse.linalg import eigsh
max_ew = np.max(eigsh(A, return_eigenvectors=False))

TI = np.where(bq.sw <= 0.70, 1.0, 0.0)

from scipy.sparse.linalg import eigsh
eigsh(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, 1000, return_eigenvectors=False)

from primate.trace import sl_trace, sl_gauss
from primate.plotting import figure_trace
tr_est = sl_trace(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, "smoothstep", orth=15, num_threads=8, atol=0.90, maxiter=250)
np.sum(tr_est != 0.0)
# np.mean(tr_est)
# 12042
show(figure_trace(tr_est))

## This can't be right, 
sl_trace(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, fun = "smoothstep", orth=15, num_threads=8, atol=0.90, maxiter=250)

def approx_matvec(A, v: np.ndarray, k: int = None, f: Callable = None):
  from scipy.linalg import eigh_tridiagonal
  from primate.diagonalize import lanczos
  k = A.shape[1] if k is None else int(k)
  (a,b), Q = lanczos(A, v, deg=k, orth=10, return_basis=True)
  # print(a,b)
  # print(v)
  rw, V = eigh_tridiagonal(a,b, eigvals_only=False)  # lanczos_quadrature(A, v, )
  rw = rw if f is None else f(rw)
  y = np.linalg.norm(v) * (Q @ V @ (V[0,:] * rw))
  return y

A = bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T
class Matvec:
  def __init__(self, A, f: Callable):
    self.A = A
    self.f = f
    self.shape = A.shape
  
  def matvec(self, v):
    return approx_matvec(A, v, k = 20, f=self.f)

# tr_est = xtrace(A, 100)[0]
# A.trace()

## Holy crap it seems to work with XTrace
from scipy.sparse.linalg import LinearOperator, aslinearoperator
sf = spectral_rank
L = aslinearoperator(Matvec(A, sf))
v = np.random.normal(size=L.shape[1])
L @ v
# from scipy.sparse.linalg import eigsh
xtrace(L, nv=1000)
# eigsh(L, v0=v)


nw = sl_gauss(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, 250, orth=15, num_threads=8)

np.sum((bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T).sum(axis=0) > 0)

# np.sum((bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T).diagonal())

xtrace(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, nv = 100)

# from scipy.linalg import lstsq
# lstsq(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, np.random.normal(size=8058))
# from scipy.linalg.interpolative import estimate_rank
# estimate_rank(bq.Lq.bm @ diags(TI) @ bq.Lq.bm.T, 0.1)

# The left-mult is 0?
bq.Lq.operator()

L = up_laplacian(S, p=1, form='lo')
L.fq = np.where(bq.sw <= 0.70, 1.0, 0.0)

A = (L @ np.eye(L.shape[1]))

np.sum(t0 > 0)
np.sum(t1 > 0)
np.sum(t2 > 0)
np.sum(t3 > 0)

np.histogram(filter_angle(S))

filter_angle(np.array(sx.faces(S,2)))


# %% Generate curve
f_min, f_max = np.min(YL[:,2]), np.max(YL[:,2])
# def restrict_filtration(angle):
#   YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)[:,2]
#   K = sx.RankFiltration(S, fixed_filter(S, YL))
# angle # vary angle from 20 degrees to 35 degrees

Filter_angle = lambda angle: fixed_filter(S, push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)[:,2])
p_filter = ParameterizedFilter(S, family=[Filter_angle(angle) for angle in np.linspace(20, 35, 5)])

sri = SpectralRankInvariant(S, family=p_filter, p=1)
sri.sieve = [[-np.inf, 0.55, 0.70, np.inf]]
sri.sift(progress=True)
sri.summarize(spectral_rank)


# %%
# K_proj = filtration(zip(YL[:, 2], S), form="rank")
# dgm = ph(K_proj, engine="cpp")
# show(figure_dgm(dgm[1]))  # should have 0.545 lifetime h1


# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(bigraded_betti)
# profile.add_function(anchors)
# profile.add_function(push_map)
# profile.enable_by_count()
# BI = bigraded_betti(S, f1, f2, p=1, xbin=15, ybin=15)
# profile.print_stats(output_unit=1e-3, stripzeros=True)


# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(rips_complex)
# profile.enable_by_count()
# r = np.quantile(normalize(patch_geodesic), 0.45)
# rips_complex(patch_geodesic, radius=r, p = 2)
# profile.print_stats(output_unit=1e-3, stripzeros=True)

## This derives form all the points so no go 
# s = splrep(degrees, g(1e-6), w=1/L.std(axis=1))
# p = figure(width=350, height=250)
# p.line(d_points, BSpline(*s)(d_points), color='green')
# p.step(degrees, np.round(g(1e-6)), color='red')
# show(p)


from pbsig.vis import figure_scatter
show(figure_scatter(np.c_[np.arange(30), spectral_sum]))

# %% Make a nice figure using ...
# from scipy.stats import gaussian_kde
gaussian_kde(X.T).evaluate(X.T) 

# Benchmark
# from pbsig.persistence import validate_decomp, generate_dgm, boundary_matrix 
# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(filtration)
# profile.add_function(ph)
# profile.add_function(boundary_matrix)
# profile.add_function(generate_dgm)
# profile.add_function(validate_decomp)
# profile.enable_by_count()
# dgm = ph(K_proj, engine='cpp')
# profile.print_stats(output_unit=1e-3, stripzeros=True)
# D = np.array([
#   [2, -1, 0, -1, 0, 0, 0, 0, 0],
#   [-1, 3, -1, 0, -1, 0, 0, 0, 0],
#   [0, -1, 2, 0, 0, -1, 0, 0, 0],
#   [-1, 0, 0, 3, -1, 0, -1, 0, 0],
#   [0, -1, 0, -1, 4, -1, 0, -1, 0],
#   [0, 0, -1, 0, -1, 3, 0, 0, -1],
#   [0, 0, 0, -1, 0, 0, 2, -1, 0],
#   [0, 0, 0, 0, -1, 0, -1, 3, -1],
#   [0, 0, 0, 0, 0, -1, 0, -1, 2]
# ])
