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
S = rips_complex(n_geodesic, radius=(connected_diam * 1.40) / 2, p = 2)

# S_dict = { p : np.array(sx.faces(S, 2)) for p in range(3) }

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
# import gudhi
# sparse_S = gudhi.RipsComplex(distance_matrix=squareform(n_geodesic), max_edge_length=(connected_diam * 1.45), sparse=0.85)
# sparse_st = sparse_S.create_simplex_tree(max_dimension=2)
# # sparse_st.num_vertices()
# print(f"Number simplices: {sparse_st.num_simplices()}")
# S = sx.simplicial_complex([s for s, fs in sparse_st.get_simplices()], form='tree')
# sf_map = { sx.Simplex(s) : fs for s,fs in sparse_st.get_simplices() }
# s_values = np.array([sf_map[s] for s in S])
# geodesic_filter = sx.fixed_filter(S, s_values)
# f2 = geodesic_filter

# K = sx.RankFiltration([(k,s) for s,k in sparse_st.get_filtration()])
# sparse_st.compute_persistence()
# sparse_st.persistence_pairs()
# S_sparse = sx.simplicial_complex([s for s, v in list(sparse_st.get_simplices())], form='rank')

# %% Compute the hilbert function 
from pbsig.rivet import figure_betti, anchors, push_map
BI = bigraded_betti(S, f1, f2, p=1, xbin=30, ybin=30)

# %% Show the hilbert function
p = figure_betti(BI, width=350, height=350, size=5) #y_range = (0.95*connected_diam, connected_diam*1.60)))
p.aspect_ratio = 1
p.toolbar_location = None
p.title = "Hilbert function of bigraded Betti numbers"
p.yaxis.axis_label = "diameter (rips)"
p.xaxis.axis_label = "codensity (kde)"
show(p)

from bokeh.models import Span, Legend, LegendItem
conn_span = Span(location=connected_diam, dimension='width', name="Connection diam")

ed = enclosing_radius(patch_geodesic) * 2

# li_cr = LegendItem(label="connected radius")
# li_er = LegendItem(label="enclosing radius")
# li_sl = LegendItem(label="silverman lb")
# li_sb = LegendItem(label="silverman ub")
# legend = Legend(items=[li_conn], title="")
legend.title_text_font_size = '10px'
legend.padding = 2
legend.margin = 2
legend.label_width = 10
legend.label_text_line_height = 0.8
# p.add_layout(legend)
show(p)


# p.line(x=[0.0, 1.0], y=[line_f(0.0, angle), line_f(1.0, angle)], color='blue')
# show(p)

# %% Choose the intercept + angle by hand 
offset, angle = 0.11, 29
# deg2slope = lambda deg: np.sin(np.deg2rad(deg))
line_f = lambda x, theta: np.tan(np.deg2rad(theta)) * x + offset

p = figure_betti(BI, width=550, height=550, show_coords=True, highlight=5, size=5, match_aspect=True) #y_range = (0.95*connected_diam, connected_diam*1.60)))
# p.line(x=[0.0, 1.0], y=[line_f(0.0, angle), line_f(1.0, angle)], color='red')
# p.aspect_ratio = 1

angle1, angle2 = 15, 42
p.line(x=[0.0, 1.0], y=[line_f(0.0, angle1), line_f(1.0, angle1)], line_dash='dashed')
p.line(x=[0.0, 1.0], y=[line_f(0.0, angle2), line_f(1.0, angle2)], line_dash='dashed')
show(p)

from bokeh.models import AnnularWedge
a, b = 0.60, 0.72
p.annular_wedge(x=0, y=offset, inner_radius=a, outer_radius=b, start_angle=np.deg2rad(angle1), end_angle=np.deg2rad(angle2), fill_color="#8888ee40")
show(p)

p.x_range = Range1d(0.15, 0.90)
p.y_range = Range1d(0.15, 0.90)
p.toolbar_location = None 
p.xaxis.visible = False
p.yaxis.visible = False
# p.aspect_ratio = (2,1)
show(p)

# %% See how many 2-cells lie inside the cell-decomposition of  
from pbsig.rivet import anchors
xg, yg = BI['x-grades'], BI['y-grades']

alpha = anchors(BI)
# len(np.unique(np.vstack([alpha, S]), axis=0))
assert all(alpha[:,0] > 0), "Not all lines have positive slope"
# np.unique(alpha, axis=0) # 
C,D = alpha[:,0], alpha[:,1]

## TODO: try to see if the number of diagrams is quite large compared to just using annealing 
## on fibered barcode space optimization 

def arrangement(anchors: np.ndarray, xg, yg):
  from shapely import LineString, MultiLineString, polygonize, get_parts, polygonize_full, Polygon, segmentize
  from shapely import LineString, multipolygons, segmentize, boundary, make_valid, envelope, union_all
  assert all(anchors[:,0] > 0), "Not all lines have positive slope"
  C,D = anchors[:,0], anchors[:,1]
  bbox = [[np.min(xg), np.max(xg)], [np.min(yg), np.max(yg)]]
  ys, ye = np.array(bbox[1])
  xs, xe = np.array(bbox[0])
  YS = C * xs - D
  YE = C * xe - D
  bx_s, bx_e = xs, xe
  by_s, by_e = np.min(YS), np.max(YE) 
  BB = LineString([(bx_s, by_s), (bx_e, by_s), (bx_e, by_e), (bx_s, by_e), (bx_s, by_s)])      ## bounding box
  LL = [LineString([[xs, ys], [xe, ye]]) for ys, ye in zip(YS, YE)]     ## lines intersecting the box 
  A = union_all([*LL, BB])
  subdivision = polygonize(get_parts(A))
  template_points = np.array([np.ravel(two_cell.centroid.xy) for two_cell in get_parts(subdivision) if two_cell.area > 1e-10])
  return subdivision, template_points

subdivision, TP = arrangement(alpha, xg, yg)

## Restricting to only points in the box of interest
in_box = np.logical_and(np.logical_and(alpha[:,0] >= 0.40, alpha[:,0] <= 0.70), np.logical_and(alpha[:,1] <= 0.50, alpha[:,1] >= 0.35))
subdivision, TP = arrangement(alpha[in_box], xg, yg)

#%% Nice picture with bokeh 
p = figure(width=400, height=400)
p.multi_line(
  xs=[line.coords.xy[0] for line in LL], 
  ys=[line.coords.xy[1] for line in LL]
)
show(p)

# %% Annealing ?
from spirit.apparent_pairs import SpectralRI
from combin import comb_to_rank

Y0 = np.c_[f1(sx.faces(S,0)), f2(sx.faces(S,0))]
Y1 = np.c_[f1(sx.faces(S,1)), f2(sx.faces(S,1))]
Y2 = np.c_[f1(sx.faces(S,2)), f2(sx.faces(S,2))]

RI = SpectralRI(n=sx.card(S,0), max_dim=2)
RI._simplices[0] = np.arange(sx.card(S, 0))
RI._simplices[1] = comb_to_rank(S.edges, order='colex', n=sx.card(S, 0))
RI._simplices[2] = comb_to_rank(S.triangles, order='colex', n=sx.card(S, 0))
RI._D[0] = RI.boundary_matrix(0)
RI._D[1] = RI.boundary_matrix(1).astype(np.float64)
RI._D[2] = RI.boundary_matrix(2).astype(np.float64)

RI.query(1, a,b,c,d, method="cholesky", summands=False)

RI._weights[0] = push_fast(Y0, a=np.tan(np.deg2rad(angle)), b=offset)
RI._weights[1] = push_fast(Y1, a=np.tan(np.deg2rad(angle)), b=offset)
RI._weights[2] = push_fast(Y2, a=np.tan(np.deg2rad(angle)), b=offset)
RI._status[0] = np.zeros(sx.card(S,0))
RI._status[1] = np.zeros(sx.card(S,1))
RI._status[2] = np.zeros(sx.card(S,2))

RI.query(1, 0.20, 0.80, method="cholesky", summands=False)
RI.query(1, 0.20, 0.80, 1.20, 1.80, method="cholesky", summands=False)

np.max(push_fast(Y2, a=np.tan(np.deg2rad(angle)), b=offset))

# %% Construct a parameterized flter over convex-combination of a bilftration
def push_fast(X, a, b):
  """Push-map from RIVET - Projects points in X to the line y = a * x + b """
  x, y = X[:,0], X[:,1]
  cond = (x * a + b) < y
  tmp_0 = np.where(cond, (y - b) / a,  x)
  tmp_1 = np.where(cond, y, a * x + b)
  return np.sqrt(tmp_0 ** 2 + (b - tmp_1) ** 2)

from pbsig.interpolate import ParameterizedFilter
Y = np.c_[f1(S), f2(S)]
p_filter = ParameterizedFilter(S)
p_filter.family = lambda theta: push_fast(Y, a=np.tan(np.deg2rad(theta)), b=offset)
p_filter.domain = (angle1, angle2)

# import timeit
# timeit.timeit(lambda: push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset), number = 5) 
# # 1.5 s
# timeit.timeit(lambda: push_fast(Y, a=np.tan(np.deg2rad(angle)), b=offset), number = 5)
# 0.002796350046992302 seconds 

# %% This should show 5 points in H1 with high persistence
from pbsig.persistence import ph 
from pbsig.vis import figure_dgm

YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)
filter_angle = fixed_filter(S, YL[:,2])
K = sx.RankFiltration(S, filter_angle)
dgms = ph(K)


#%% First, verify we get 5 in the spectral rank invariant
from primate.trace import hutch
from primate.functional import numrank
from pbsig.betti import BettiQuery, betti_query

bq = BettiQuery(S, p=1)
bq.sign_width = 0.0
bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="simple")  # hutch(L, maxiter=5)
bq.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="simple")
# t0, t1, t2, t3 = list(bq.generate(i=a, j=b, mf=lambda x: x))[0]
bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap=1e-2) 
bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="auto") 
bq.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="auto") 

bq.weights[0] = p_filter(angle, 0)
bq.weights[1] = p_filter(angle, 1)
bq.weights[2] = p_filter(angle, 2)

bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="simple")

## test it out with angle 
bq(i=0.60, j=0.65, terms=False)
bq(i=0.40, j=0.60, k=0.65, l=0.80, terms=False)

# numrank(bq.Lq.operator(deflate=True), atol=0.50, confidence=0.90, gap="auto")
# 207 - 477 - 1973 + 2248

## Try to compute rank across a prameter range
bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="simple")  # hutch(L, maxiter=5)
bq.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else numrank(L, atol = 0.50, confidence=0.90, gap="simple")
for theta in np.linspace(angle1, angle2, 20):
  bq.weights[0] = p_filter(theta, 0)
  bq.weights[1] = p_filter(theta, 1)
  bq.weights[2] = p_filter(theta, 2)
  num_points = bq(i=a, j=b, terms=False)
  if num_points > 10: 
    assert False 
  print(num_points)

## Try the Tikhonov
tikhonov = lambda x: x / (x + 1e-2)
bq.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else hutch(L, fun=tikhonov, atol=1.5, maxiter=3000)  # hutch(L, maxiter=5)
bq.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else hutch(L, fun=tikhonov, atol=1.5, maxiter=3000)
tik_results = np.zeros(shape=(10, 100))
for ii in range(tik_results.shape[0]):
  for jj, theta in enumerate(np.linspace(angle1, angle2, tik_results.shape[1])):
    bq.weights[0] = p_filter(theta, 0)
    bq.weights[1] = p_filter(theta, 1)
    bq.weights[2] = p_filter(theta, 2)
    tik_results[ii,jj] = bq(i=a, j=b, terms=False)
  print(ii)

## Plot the results alongside the angular wedge 
R = np.abs(5 - tik_results)
q = figure(width=450, height = 250)
angles = np.linspace(angle1, angle2, tik_results.shape[1])
for ii in range(R.shape[0]):
  q.line(angles, R[ii,:], line_width=0.50, line_color="#0096FF6a")

q.line(angles, R.mean(axis=0), line_width=1.0, line_color="#FF0000a8")
show(q)

# %% Make row plot of hilbert wedge + diagram + objective 
from bokeh.layouts import row
p.width = 400
p.height = q.height = 250
q.width = 250
show(row(p, q))







## What do we get out of it? If we optimize, we get an angle, a 1-D filtration, with 5 persistent point in the box we wanted 
## What do we get at the manifold level? 
## Any type of projection we can show? we get a filtration in which the 5 circle's are the most apparent 
## we get a linear combination of simplex codensity / geodesic rips wherein the topology is 
## TODO simulated annelaing 

## Maybe get the homology cocycles and show them?


# f_ls = sx.lower_star_filter(p_filter(23, 0))
# test1 = f_ls(S)
# np.max(np.abs(p_filter(23) - test1))

# %% Look at the generators
from pbsig.persistence import generate_dgm
R,V = ph(K, output="RV")
dgms = generate_dgm(K, R, simplex_pairs=True)
show(figure_dgm(dgms[1]))

top_5 = np.argpartition(dgms[1]['death'] - dgms[1]['birth'], kth=5)[-5:]
dgms[1]['creators'][top_5]
dgms[1]['destroyers'][top_5]


import combin 
creators = [sx.Simplex([s[1], s[2]]) for s in dgms[1]['creators'][top_5]]
destroyers = [sx.Simplex([s[1], s[2], s[3]]) for s in dgms[1]['destroyers'][top_5]]
creator_indices = []
for ii, (fv, s) in enumerate(K):
  if s in creators:
    creator_indices.append(ii)

def generator_vertices(K, V, c: int):
  from more_itertools import first_true, unique_everseen
  cycle_edges = np.array([K[i][1] for i in V[:,c].indices])
  edges = set([tuple(sorted(e)) for e in cycle_edges])
  cycle = [edges.pop()]
  while len(cycle) != len(cycle_edges):
    i, j = cycle[-1]
    k, l = first_true(edges, pred=lambda e: (e[0] in (i,j)) or (e[1] in (i,j)))
    assert k == j or i == l or i == k or j == l
    cycle.extend([(k, l)])
    edges -= set([(k,l)])
  vertices = np.array(list(unique_everseen(np.ravel(cycle))))
  return vertices


Klein.basis = "natural"

# 94, 132, 135, 262, 334
vertices = generator_vertices(K, V, 334)
generator_patches = Klein.patch_data[stratified_ind[vertices],:]

p = figure(width=len(vertices) * 100, height=100)
for i, patch in enumerate(generator_patches):
  patch = Klein.patch_rgba(patch)
  p.image_rgba([patch], x=i*12, y=0, dw=10, dh=10)
p.xaxis.visible = False
p.yaxis.visible = False
show(p)

## Fix the box 
a,b = 0.65, 0.82
print(S)
print(f"Point in box ({a}, {b}): { np.sum(np.logical_and(dgms[1]['birth'] <= a, dgms[1]['death'] > b)) }")


# %% 

from pbsig.csgraph import deflate_sparse
A = deflate_sparse(bq.Lq.operator()).tocsr()
hutch(A, fun=tikhonov, atol=1, maxiter=3000)



## Show the truth
filter_angle = fixed_filter(S, p_filter(theta))
K = sx.RankFiltration(S, filter_angle)
dgms = ph(K)
show(figure_dgm(dgms[1]))

print(S)
print(f"Point in box ({a}, {b}): { np.sum(np.logical_and(dgms[1]['birth'] <= a, dgms[1]['death'] > b)) }")

bq(i=a, j=b, mf=lambda x: x, terms=True)

from pbsig.linalg import PsdSolver
solver = PsdSolver()
A = bq.operator(3, a, b)
ew = solver(A)
np.sum(solver(A) >= 1e-4)

# from pbsig.csgraph import deflate_sparse
# from primate.functional import RelativeErrorBound
# from primate.diagonalize import lanczos
# from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
# import bisect
# A = deflate_sparse(A)
# EPS = np.finfo(A.dtype).eps
# deg = 20
# n = A.shape[0]
# re_bnd = RelativeErrorBound(n)
# deg_bound = max(bisect.bisect_left(re_bnd, -0.01) + 1, deg)
# a,b = lanczos(A, deg=deg_bound)
# rr, rv = eigh_tridiagonal(a,b)
# tol = np.max(rr) * A.shape[0] * EPS # NumPy default tolerance 

## This is likely wrong 
min_id = np.flatnonzero(rr >= tol)[np.argmin(rr[rr >= tol])]
coeff = min([b[min_id-1], b[min_id], b[min_id+1]])
gap = rr[min_id] - coeff * np.abs(rv[-1,min_id])
gap = rr[min_id] if gap < 0 else gap

## Fourth term!
# 254 - 75 - 317 + 143

254 - 75 - 248 + 158


from scipy.sparse.linalg import eigsh 
np.sum(eigsh(bq.Lp.operator(), k = 248, return_eigenvectors=False) >= 1e-2)


tikhonov = lambda x: x / (x + 1e-2)
mf = lambda L: hutch(L, fun=tikhonov, rtol=0.10, maxiter=300)
bq.q_solver = mf
bq.p_solver = mf
t0, t1, t2, t3 = list(bq.generate(i=a, j=b, mf=np.sum))[0]

print(f"{np.sum(t0)}, {t1}, {t2}, {t3}")
np.sum(t0) - t1 - t2 + t3

## 

p_filter = ParameterizedFilter(S)
p_filter(0.0)




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
