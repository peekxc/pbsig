# %% Imports
from typing import *

import numpy as np
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
from splex.filters import *
from scipy.io import loadmat

output_notebook()

# %% data 
dct_patch_data = loadmat("/Users/mpiekenbrock/Downloads/n50000Dct.mat")
dct_patches = dct_patch_data['n50000Dct']
dct_basis = [None]*8
dct_basis[0] = (1/np.sqrt(6)) * np.array([1,0,-1]*3).reshape(3,3)
dct_basis[1] = dct_basis[0].T
dct_basis[2] = (1/np.sqrt(54)) * np.array([1,-2,1]*3).reshape(3,3)
dct_basis[3] = dct_basis[2].T
dct_basis[4] = (1/np.sqrt(8)) * np.array([[1,0,-1],[0,0,0],[-1,0,1]])
dct_basis[5] = (1/np.sqrt(48)) * np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
dct_basis[6] = (1/np.sqrt(48)) * np.array([[1,-2,1],[0,0,0],[-1,2,-1]])
dct_basis[7] = (1/np.sqrt(216)) * np.array([[1,-2,-1],[-2,-4,-2],[1,-2,1]])
dct_basis_mat = np.hstack([np.ravel(b)[:,np.newaxis] for b in dct_basis])

## Projection onto the DCT basis
from pbsig.linalg import pca
grayscale_patches = dct_patches @ dct_basis_mat.T
pca_patches = pca(grayscale_patches)

p = figure(width=300, height=300)
p.scatter(*pca_patches.T, color = "#808080a0", size=2)

def figure_img_patch(p: figure, ind: int, w: float = 0.10, h: float = 0.10):
  lb, ub = grayscale_patches.min(), grayscale_patches.max()
  normalize = lambda x: (x - lb) / (ub - lb)
  patch = grayscale_patches[ind]
  img = np.empty((3,3), dtype=np.uint32)
  view = img.view(dtype=np.uint8).reshape((3,3, 4))
  for cc, (i,j) in enumerate(product(range(3), range(3))):
    view[i, j, 0] = int(255 * normalize(patch[cc]))
    view[i, j, 1] = int(255 * normalize(patch[cc]))
    view[i, j, 2] = int(255 * normalize(patch[cc]))
    view[i, j, 3] = 255
  patch_x, patch_y = pca_patches[ind]
  p.image_rgba(image=[img], x=patch_x-w, y=patch_y-h, dw=2*w, dh=2*h)
show(p)

from bokeh.io import curdoc
curdoc().clear()
from bokeh.palettes import gray
import datashader as ds, pandas as pd
p = figure(width=600, height=600)
canvas = ds.Canvas(int(300*3.5), int(300*3.5), x_range=(-1, 1), y_range=(-1, 1))
patch_df = pd.DataFrame(dict(x=pca_patches[:,0], y=pca_patches[:,1]))
agg = canvas.points(patch_df, x="x", y="y")
im = ds.transfer_functions.shade(agg, cmap=gray(256), min_alpha=100, rescale_discrete_levels=False)
# im = ds.transfer_functions.spread(im)
im = ds.transfer_functions.dynspread(im, threshold=0.95)
p.image_rgba(image=[im.to_numpy()], x=-1, y=-1, dw=2, dh=2, dilate=True)
show(p)

from pbsig.shape import landmarks
landmark_ind, _ = landmarks(pca_patches, 20)
for i in landmark_ind:
  figure_img_patch(p, i)
show(p)

# %% Stratified sampling 
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma

## Get a KNN density estimate
patch_kdtree = KDTree(grayscale_patches)
knn_dist, knn_ind = patch_kdtree.query(grayscale_patches, k=15)
vol_ball = lambda n, R: np.pi**(n/2) * np.reciprocal(gamma(n/2 + 1)) * R**n
N = grayscale_patches.shape[1]
knn_density = (15 - 1) / N * vol_ball(8, knn_dist.max(axis=1))

## Do proportionate allocation sampling
def proportional_integers(x: np.ndarray, n: int):
  """Given an array 'x', find an array of integers 'y' such that sum(y) = n and the x[i]/x[j] ~= y[i]/y[j]"""
  scaling_factor = n / sum(x)
  y = np.round(np.array(x) * scaling_factor).astype(int)
  while sum(y) != n: 
    idx = np.argmax(x - y) if sum(y) < n else np.argmax(y) 
    y[idx] += 1 if sum(y) < n else -1
  return y

def stratify_sample(n: int, f: np.ndarray, n_strata: int = 25, return_strata: bool = False):
  """Stratify samples _n_ indices in [0, len(f)-1] using proportionate allocation."""
  assert isinstance(f, np.ndarray), "f must be an array"
  from array import array
  from itertools import pairwise
  bin_counts, bin_edges = np.histogram(f, bins=n_strata)
  sample_counts = proportional_integers(bin_counts, n)
  indices = {} if return_strata else array('I') 
  for cc, ((a,b), ns) in enumerate(zip(pairwise(bin_edges), sample_counts)):
    stratum_ind = np.flatnonzero(np.logical_and(a <= f, f <= b))
    sample_ind = np.random.choice(stratum_ind, size=ns, replace=False)
    if return_strata:
      indices[cc] = dict(n=ns, stratum=stratum_ind, interval=(a,b))
    else:
      indices.extend(sample_ind)
  return indices if return_strata else np.array(indices)

from pbsig.csgraph import neighborhood_graph
from scipy.sparse.csgraph import floyd_warshall
# stratified_ind = stratify_sample(50, knn_density)

from more_itertools import collapse
knn_strata = stratify_sample(300, knn_density, return_strata = True)
strata = []
for k, v in knn_strata.items():
  v_str = v['stratum']
  if v['n'] > 0: 
    l_ind, _ = landmarks(grayscale_patches[v_str,:], k=v['n'])
    strata.append(v_str[l_ind])
stratified_ind = np.array(list(collapse(strata)))


for i in stratified_ind:
  figure_img_patch(p, i, w=0.025, h=0.025)
show(p)


patch_knn_graph = neighborhood_graph(grayscale_patches[stratified_ind], k = 15, weighted=True)
patch_geodesic = squareform(floyd_warshall(patch_knn_graph.tocsr(), directed=False))

#%% Form the two filter functions to create the bifiltration of
from splex import flag_filter, lower_star_filter
normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
sample_codensity = max(knn_density[stratified_ind]) - knn_density[stratified_ind]
codensity_filter = lower_star_filter(normalize(sample_codensity))
geodesic_filter = flag_filter(normalize(patch_geodesic))

# %% Call rivet_console to get the minimal presentation
# bigraded_betti 
from pbsig.vis import figure_complex, show
from splex import rips_complex
print(f"Is connected? {not(np.any(np.isinf(patch_geodesic)))}")
f1 = codensity_filter
f2 = geodesic_filter
S = rips_complex(patch_geodesic, radius=np.quantile(normalize(patch_geodesic), 0.25), p = 2)
#show(figure_complex(S, pos=pca_patches[stratified_ind,:]))

# %% 
from pbsig.rivet import figure_betti
BI = bigraded_betti(S, f1, f2, p=1, xbin=25, ybin=25, input_file="bifilt.txt", output_file="betti_out.txt")
show(figure_betti(BI))




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
