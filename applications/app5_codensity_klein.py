import numpy as np
from itertools import product
from typing import * 
from scipy.sparse import spmatrix
from numpy.typing import ArrayLike
from bokeh.io import output_notebook, export_png
from bokeh.plotting import figure, show
from bokeh.io import export_png
from pbsig.utility import progressbar
from more_itertools import chunked
from pbsig.vineyards import linear_homotopy, transpose_rv
from math import comb
from pbsig.utility import progressbar
from pbsig.vis import bin_color
from pbsig.vis import figure_complex
output_notebook()

from scipy.io import loadmat
dct_patch_data = loadmat("/Users/mpiekenbrock/Downloads/n50000Dct.mat")
dct_patches = dct_patch_data['n50000Dct']

D = np.array([
  [2, -1, 0, -1, 0, 0, 0, 0, 0],
  [-1, 3, -1, 0, -1, 0, 0, 0, 0],
  [0, -1, 2, 0, 0, -1, 0, 0, 0],
  [-1, 0, 0, 3, -1, 0, -1, 0, 0],
  [0, -1, 0, -1, 4, -1, 0, -1, 0],
  [0, 0, -1, 0, -1, 3, 0, 0, -1],
  [0, 0, 0, -1, 0, 0, 2, -1, 0],
  [0, 0, 0, 0, -1, 0, -1, 3, -1],
  [0, 0, 0, 0, 0, -1, 0, -1, 2]
])

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

def figure_img_patch(p: figure, ind: int):
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
  p.image_rgba(image=[img], x=patch_x-0.10, y=patch_y-0.10, dw=0.20, dh=0.20)
show(p)

from bokeh.io import curdoc
curdoc().clear()
from bokeh.palettes import gray
import datashader as ds, pandas as pd
p = figure(width=600, height=600)
canvas = ds.Canvas(int(300*1.25), int(300*1.25), x_range=(-1, 1), y_range=(-1, 1))
patch_df = pd.DataFrame(dict(x=pca_patches[:,0], y=pca_patches[:,1]))
agg = canvas.points(patch_df, x="x", y="y")
im = ds.transfer_functions.shade(agg, cmap=gray(256))
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
knn_density = (300 - 1) / N * vol_ball(8, knn_dist.max(axis=1))

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
stratified_ind = stratify_sample(50, knn_density)
patch_knn_graph = neighborhood_graph(grayscale_patches[stratified_ind], k = 15, weighted=True)
patch_geodesic = floyd_warshall(patch_knn_graph.tocsr())

#%% Form the two filter functions to create the bifiltration of
from splex import flag_weight, lower_star_weight
sample_codensity = -(max(knn_density[stratified_ind]) - knn_density[stratified_ind])
codensity_filter = lower_star_weight(sample_codensity)
geodesic_filter = flag_weight(patch_geodesic)

# %% Call rivet+console to get the minimal presentation
# bigraded_betti 
from pbsig.vis import figure_complex, show
from splex import rips_complex
print(f"Is connected? {not(np.any(np.isinf(patch_geodesic)))}")
f1 = codensity_filter
f2 = geodesic_filter
S = rips_complex(patch_geodesic, radius=np.quantile(patch_geodesic, 0.80), p = 2)
#show(figure_complex(S, pos=pca_patches[stratified_ind,:]))


# %% 2-d persistence
from splex.filtrations import RankFiltration
from pbsig.datasets import noisy_circle
from scipy.spatial.distance import pdist
np.random.seed(1234)
X = noisy_circle(35, n_noise=10)
S = rips_complex(X, radius = np.quantile(pdist(X), 0.45)/2, p = 2)
show(figure_complex(S, X))

vertex_density = gaussian_kde(X.T).evaluate(X.T) 
codensity_filter = lower_star_weight(max(vertex_density) - vertex_density)
diameter_filter = flag_weight(pdist(X))
f1,f2 = codensity_filter, diameter_filter

inp_file_str = "--datatype bifiltration\n"
inp_file_str += "--xlabel density\n"
inp_file_str += "--ylabel diameter\n"
inp_file_str += "\n"
for s, fs_1, fs_2 in zip(S, f1(S), f2(S)):
  inp_file_str += f"{' '.join([str(v) for v in s])} ; {fs_1} {fs_2}\n"

import subprocess
from tempfile import NamedTemporaryFile
tf = NamedTemporaryFile()
with open(tf.name, "w") as inp_file:
  inp_file.write(inp_file_str)
output_tf = NamedTemporaryFile()
# inp_file_str = f"-H {H} --xbins {xbin} --ybins {ybin}\n"

import pathlib
H, xbin, ybin = 1, 15, 15
xreverse, yreverse = True, False
rivet_path = pathlib.Path("/Users/mpiekenbrock/rivet/rivet_console").resolve()
assert rivet_path.exists(), "Did not find path to 'rivet_console'"
rivet_path_str = str(rivet_path.absolute())
rivet_cmd = [rivet_path_str, inp_file.name]
rivet_cmd += ["--betti", "--homology", str(H), "--xbins", str(xbin), "--ybins", str(ybin)]
# rivet_cmd += ["--xreverse" if xreverse else ""]
# rivet_cmd += ["--yreverse" if yreverse else ""]
rivet_cmd += [">", output_tf.name]
res = subprocess.call(' '.join(rivet_cmd), shell=True)

with open(output_tf.name) as f:
  rivet_out_lines = [line.rstrip('\n') for line in f]

xi, yi, di, bi = [rivet_out_lines.index(key) for key in ['x-grades', 'y-grades', 'Dimensions > 0:', 'Betti numbers:']]
b0,b1,b2 = [rivet_out_lines.index(key) for key in ['xi_0:', 'xi_1:', 'xi_2:']]
betti_info = {}
betti_info['x-grades'] = np.array([eval(x) for x in rivet_out_lines[(xi+1):yi] if len(x) > 0])
betti_info['y-grades'] = np.array([eval(x) for x in rivet_out_lines[(yi+1):di] if len(x) > 0])
betti_info['hilbert_pts'] = [eval(x) for x in rivet_out_lines[(di+1):bi] if len(x) > 0]
betti_info['betti_0'] = [eval(x) for x in rivet_out_lines[(b0+1):b1] if len(x) > 0]
betti_info['betti_1'] = [eval(x) for x in rivet_out_lines[(b1+1):b2] if len(x) > 0]
betti_info['betti_2'] = [eval(x) for x in rivet_out_lines[(b2+1):] if len(x) > 0]

## Show the Hilbert function 
xg, yg = betti_info['x-grades'], betti_info['y-grades']
betti0 = np.array([(xg[i],yg[j]) for i,j,val in betti_info['betti_0']])
betti1 = np.array([(xg[i],yg[j]) for i,j,val in betti_info['betti_1']])
hilbert = np.array([(xg[i],yg[j],val) for i,j,val in betti_info['hilbert_pts']])

## basic 
from pbsig.vis import rgb_to_hex
p = figure(width=250, height=250)
p.scatter(*betti0.T, size=10, color=rgb_to_hex([52, 209, 93, int(0.40*255)]))
p.scatter(*betti1.T, size=10, color=rgb_to_hex([212, 91, 97, int(0.40*255)]))
p.scatter(*hilbert[:,:2].T, size=10, marker='x', color=np.array(['gray', 'blue'])[hilbert[:,2].astype(int)-1])
show(p)


# %% Show the persistence diagram for a specific combination to optimize for
offset = -0.0109
angle = 85.4741

