# %%
import numpy as np
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

# %% Load data 
from pbsig.datasets import noisy_circle
from splex.geometry import delaunay_complex
np.random.seed(1234)
X = noisy_circle(150, n_noise=20, perturb=0.15) ## 80, 30, 0.15 
S = delaunay_complex(X)

from pbsig.csgraph import WeightedLaplacian
L = WeightedLaplacian(S, p = 1)
L = WeightedLaplacian(S, p = 0)

# %% 
show(figure_complex(S, pos=X, width=300, height=300))

# %% Create filtering function
from scipy.stats import gaussian_kde
def codensity(bw: float, normalize: bool = True):
  x_density = gaussian_kde(X.T, bw_method=bw).evaluate(X.T)
  x_codensity = max(x_density) - x_density
  return x_codensity / np.sum(x_codensity) if normalize else x_codensity

# %% Evaluate H1-persistence across a 1-parameter family
from splex import lower_star_filter
from pbsig.interpolate import ParameterizedFilter
bw_scott = gaussian_kde(X.T, bw_method='scott').factor
bw_bnds = (bw_scott*0.10, bw_scott*2)
timepoints = np.linspace(*bw_bnds, 100) # np.geomspace(*bw_bnds, 100)
codensity_family = ParameterizedFilter(S, family = [lower_star_filter(codensity(bw)) for bw in timepoints])
# p_family.interpolate(interval=bw_bnds)

# %% 
from pbsig.betti import SpectralRankInvariant
ri = SpectralRankInvariant(S, family = codensity_family, p = 1, normed = True, isometric=False)

# %% Compute all the diagrams in the family
from pbsig.vis import figure_vineyard
dgms = ri.compute_dgms(S, progress=True)
vineyard_fig = figure_vineyard(dgms, p = 1)

# %% Basically anything in [0, 0.01] x [0.02, 0.03] works
# ri.sieve = np.array([[0.01, 0.015, 0.02, 0.025]])
ri.sieve = np.array([[-np.inf, 0.015, 0.02, +np.inf]])
show(ri.figure_sieve(figure=vineyard_fig, fill_alpha=0.15))

# %% Look at the numerical rank at a fixed threshold (that tends to work)
from pbsig.linalg import spectral_rank
filter_rngs = np.array([np.max(f(S)) for f in ri.family])
w = np.max(filter_rngs) / 2
ri.sift(w=w) # w is a smoothing parameter

# %% Try multiple spectral functions
from pbsig.linalg import spectral_rank
show(ri.figure_summary(func = spectral_rank))
show(ri.figure_summary(func = lambda x: x**2))

## How do these two not match? 
np.array([betti_query(S, f, matrix_func = spectral_rank, p = 1, i = 0.015, j = 0.02) for f in ri.family])

from more_itertools import nth
from pbsig.betti import betti_query
f = nth(ri.family, 0)

from pbsig.csgraph import WeightedLaplacian
L = WeightedLaplacian(S, p = 1, normed=True, form='lo')
L.reweight()
np.array(L.op.fpl)
np.array(L.op.fpr)
np.array(L.op.fq)
solver(L.operator())
np.array(L.simplex_weights)

# %% 
betti_query(S, f, matrix_func = lambda x: x, i = 0.015, j = 0.02, w=w, terms = True, form='array', normed=True)
# f1 = nth(ri.family, 46)
# f2 = nth(ri.family, 47)

# np.max(np.abs(f1(S) - f2(S)))



# %% Plot 
from bokeh.layouts import row, column
vineyard_fig = figure_vineyard(dgms, p = 1)
vineyard_fig.line(x=[-10, 0.015], y=[0.02, 0.02], color='gray', line_dash='dashed', line_width=2) 
vineyard_fig.line(x=[0.015, 0.015], y=[0.02, 10], color='gray', line_dash='solid', line_width=2) 
vineyard_fig.rect(x=0.015-5, y=0.02+5, width=10, height=10, fill_color='gray', fill_alpha=0.20, line_width=0.0)
vineyard_fig.scatter(x=0.015, y=0.02, color='red', size=8)
show(vineyard_fig)

def tikhonov(eps: float):
  return lambda x: np.sum(x / (x + eps))

def tikhonov_smoothed(lb: float, ub: float, n: int = 32):
  eps = np.geomspace(lb, ub, n)
  def _mean_logspaced(x):
    return np.sum((np.tile(x[:,np.newaxis], n) * np.reciprocal(x[:,np.newaxis] + eps)).mean(axis=1))
  return _mean_logspaced

def center(x: np.ndarray, ref: np.ndarray):
  # return x - np.min(x) 
  return x + np.mean(ref - x)

def convolve_sg(**kwargs):
  from scipy.signal import savgol_filter
  sg_kwargs = dict(window_length=5, polyorder=3) | kwargs
  def _sg(x: np.ndarray):
    return savgol_filter(x, **sg_kwargs)
  return _sg

# x = center(np.ravel(ri.summarize(tikhonov_smoothed(1e-9, 1e-2, 30))))
sg = convolve_sg(window_length=7, polyorder=5, deriv=0, delta=0.1)

def heat_trace(eps):
  def _trace(x):
    return 1.0 - np.exp(-x*(1/eps))
  return _trace

## Smoothed tikhonov
ref_betti = np.ravel(ri.summarize(spectral_func=spectral_rank).astype(int))
p = figure(width=350, height=300)
p.step(timepoints, ref_betti, color='black')
p.line(timepoints, center(sg(np.ravel(ri.summarize(tikhonov_smoothed(1e-9, 1e-2, 30)))), ref_betti), color='blue')
p.line(timepoints, center(sg(np.ravel(ri.summarize(tikhonov_smoothed(1e-6, 1e-2, 30)))), ref_betti), color='red')
p.line(timepoints, center(sg(np.ravel(ri.summarize(tikhonov_smoothed(1e-4, 1e-1, 30)))), ref_betti), color='orange')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(tikhonov_smoothed(1e-3, 0.1, 30)))), ref_betti), color='green')
show(p)

## Heat trace
from pbsig.linalg import heat
heat_rev = lambda x: 1.0 - heat(x, t=1/100.0)
ref_betti = np.ravel(ri.summarize(spectral_func=heat_rev).astype(int))
p = figure(width=350, height=300)
p.line(timepoints, ref_betti, color='black')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(heat_trace(1e-2)))), ref_betti), color='blue')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(heat_trace(0.005)))), ref_betti), color='green')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(heat_trace(1e-3)))), ref_betti), color='red')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(heat_trace(1e-4)))), ref_betti), color='orange')
# p.line(timepoints, center(sg(np.ravel(ri.summarize(tikhonov_smoothed(1e-3, 0.1, 30)))), ref_betti), color='green')
show(p)

## Regular tikhonov
## TODO: fix the discontinuities!
p = figure(width=350, height=300)
p.step(timepoints, ref_betti, color='black')
p.line(timepoints, center(np.ravel(ri.summarize(tikhonov(eps=1e-9))), ref_betti), color='blue')
p.line(timepoints, center(np.ravel(ri.summarize(tikhonov(eps=1e-4))), ref_betti), color='green')
p.line(timepoints, center(np.ravel(ri.summarize(tikhonov(eps=1e-2))), ref_betti), color='red')
p.line(timepoints, center(np.ravel(ri.summarize(tikhonov(eps=1e-1))), ref_betti), color='orange')
p.line(timepoints, center(np.ravel(ri.summarize(tikhonov(eps=0.05))), ref_betti), color='purple')
show(p)

p = figure(width=350, height=300)
p.line(timepoints, np.ravel(ri.summarize(np.mean)))
show(p)


from pbsig.betti import persistent_betti
persistent_betti(S, )

## Nuclear norm 
p = figure(width=350, height=300)
p.line(timepoints, sg(center(np.ravel(ri.summarize(np.sum)), ref_betti)), color='blue')
show(p)


show(figure_complex(S, pos=X, width=300, height=300))


# %% Debugging 
# np.flatnonzero(ri.summarize(spectral_rank) != 0)
# from pbsig.betti import betti_query
# f = list(ri.family)[32]
# b = list(betti_query(S, f, matrix_func=spectral_rank, p=1, i=0.015, j=0.02, terms = True))
# b = np.ravel(np.array(b))
# b = [list(betti_query(S, f, matrix_func=spectral_rank, p=1, i=0.015, j=0.02, terms = False)) for f in ri.family]
# b = np.ravel(np.array(b))
# c = ri.summarize(spectral_rank).astype(int)
# wrong_ind = np.argmax(np.abs(ri.summarize(spectral_rank).astype(int) - b))
# wrong_ind
# num_rank_f = np.vectorize(lambda x: 1.0 if np.abs(x) > 1e-6 else 0.0)
# ri.summarize(num_rank_f)
# show(ri.figure_summary(func = num_rank_f))
# show(ri.figure_summary(func = np.sum))
# len(np.ravel(ri.summarize()))
# np.max(ri.spectra[0]['eigenvalues'])

# %% Try multiple summary functions 



# %% See the bottleneck in computing diagrams 
def all_dgms():
  for filter_f in P:
    K = filtration(S, f=filter_f, form="rank")
    dgm = ph(K, output="dgm", engine="dionysus")
  return 0 

import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(filtration)
profile.add_function(ph)
profile.add_function(boundary_matrix)
profile.add_function(generate_dgm)
profile.add_function(validate_decomp)
profile.add_function(all_dgms)
profile.enable_by_count()
all_dgms()
profile.print_stats(output_unit=1e-3, stripzeros=True)


import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(ri.sift)
profile.enable_by_count()
ri.sift()
profile.print_stats(output_unit=1e-3, stripzeros=True)


# %% Check Parameterized lapalacina works 
# from pbsig.linalg import ParameterizedLaplacian
# P_laplacian = ParameterizedLaplacian(S, family = P, p = 0)

# [L for L in P_laplacian]



# %% Optimize via e.g. ISTA 

