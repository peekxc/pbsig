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

## Get a KNN density estimate
k = 30
Klein.basis = "dct"
patch_kdtree = KDTree(Klein.patch_data)
knn_dist, knn_ind = patch_kdtree.query(Klein.patch_data, k=k)
vol_ball = lambda d: np.pi**(d/2) * np.reciprocal(gamma(d/2 + 1))
d = Klein.patch_data.shape[1]
knn_density = ((k - 1) / d) * np.reciprocal(vol_ball(8) * knn_dist.max(axis=1))

from pbsig.csgraph import neighborhood_graph
from scipy.sparse.csgraph import floyd_warshall
from pbsig.shape import landmarks, stratify_sample
# stratified_ind = stratify_sample(50, knn_density)
# density_threshold = np.quantile(knn_density, 0.95)
# knn_density[knn_density >= density_threshold] = density_threshold

from more_itertools import collapse
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

# %% 
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
S = rips_complex(normalize(patch_geodesic), radius=(connected_diam * 1.5)/2, p = 2)

# %% Compute the hilbert function 
from pbsig.rivet import figure_betti, anchors, push_map
BI = bigraded_betti(S, f1, f2, p=1, xbin=25, ybin=25)
p = figure_betti(BI, width=350, height=350, highlight=5, size=5) #y_range = (0.95*connected_diam, connected_diam*1.60)))
show(p)

# %% Choose the intercept + angle by hand 
offset, angle = 0.11, 29
line_f = lambda x: np.tan(np.deg2rad(angle)) * x + offset
p = figure_betti(BI, width=350, height=350, show_coords=True, highlight=5, size=5) #y_range = (0.95*connected_diam, connected_diam*1.60)))
p.line(x=[0.0, 1.0], y=[line_f(0.0), line_f(1.0)], color='red')
show(p)

# %% Apply the push map to get a single filtration
Y = np.c_[f1(S), f2(S)]
YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)

# %% Generate curve
f_min, f_max = np.min(YL[:,2]), np.max(YL[:,2])
def restrict_filtration(angle):
  YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)[:,2]
  K = sx.RankFiltration(S, fixed_filter(S, YL))

SpectralRankInvariant(S, family=)

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
