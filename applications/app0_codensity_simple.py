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
X = noisy_circle(150, n_noise=35, perturb=0.15, r=0.1) ## 80, 30, 0.15 
S = delaunay_complex(X)

# from pbsig.csgraph import WeightedLaplacian
# L = WeightedLaplacian(S, p = 1)
# L = WeightedLaplacian(S, p = 0)

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
ri = SpectralRankInvariant(S, family = codensity_family, p = 1, normed = False, isometric=False)

# %% Compute all the diagrams in the family
from pbsig.vis import figure_vineyard
dgms = ri.compute_dgms(S, progress=True)
vineyard_fig = figure_vineyard(dgms, p = 1)

# %% Basically anything in [0, 0.01] x [0.02, 0.03] works
# ri.sieve = np.array([[0.005, 0.01, 0.015, 0.02]])
ri.sieve = np.array([[-np.inf, 0.01, 0.015, +np.inf]])
show(ri.figure_sieve(figure=vineyard_fig, fill_alpha=0.15))

# %% Look at the numerical rank at a fixed threshold (that tends to work)
from pbsig.linalg import spectral_rank
filter_rngs = np.array([np.max(f(S)) for f in ri.family])
w = 0.1
ri.sift(w=w*np.max(filter_rngs)) # w is a smoothing parameter
assert ri.q_laplacian.sign_width == w

# %% Try multiple spectral functions
from pbsig.linalg import spectral_rank, tikhonov, heat, nuclear_interp
def convolve_sg(**kwargs):
  from scipy.signal import savgol_filter
  sg_kwargs = dict(window_length=5, polyorder=3) | kwargs
  def _sg(x: np.ndarray):
    return savgol_filter(x, **sg_kwargs)
  return _sg
sg = convolve_sg(window_length=21, polyorder=5)
scale_unit = lambda x: x if np.allclose(x, 0.0, atol=1e-7) else (x / np.abs(x).max()) + (np.abs(x).min() / np.abs(x).max())

tik_figures = []
for w in [0.1, 1.0, 10.0]:
  ri.sift(w=w*np.max(filter_rngs)) # w is a smoothing parameter
  if w == 10.0: 
    eps_rng = [1e-16, 1e-12, 1e-9, 1e-6, 1e-3]# np.geomspace(1e-3, 10, 8)
  elif w == 1.0: 
    eps_rng = [1e-12, 1e-9, 1e-6, 1e-3, 1.0]
  else: 
    eps_rng = [1e-9, 1e-6, 1e-3, 1.0, 10.0]
  sample_index = np.arange(0, len(ri.family))
  p = figure(width=300, height=300)
  p.step(sample_index / len(sample_index), np.ravel(ri.summarize(spectral_rank)), line_color='black', line_width=1.5, legend_label="0")
  for (i, eps), l_color in zip(enumerate(eps_rng), ['purple', 'blue', 'green', 'orange', 'red']):
    ri_summary = np.ravel(sg(scale_unit(ri.summarize(tikhonov(eps=eps, truncate="numrank")))))
    p.line(sample_index / len(sample_index), ri_summary, line_color=l_color, legend_label=f"{eps:.0e}", line_width=1.5)
  p.legend.title = "Smoothing"
  p.legend.location = "top_right"
  p.legend.label_text_font_size = '14px'
  p.legend.label_height = 2
  p.legend.spacing = 1
  p.legend.padding = 2
  p.legend.glyph_height = 12
  p.xaxis.axis_label = r"$$\text{Bandwidth ( } \alpha \text{ )}$$"
  p.xaxis.axis_label_text_font_size = '16px'
  # p.title = r"$$\text{Tikhonov reg. RI ( }\omega" + f" = {w})" + r"$$" #r"$$\lVert f \rVert_\infty$$"
  p.title = f"Tikhonov reg. RI ( w = {w})"
  p.title.align = 'center'
  p.title.text_font_size = '20px'
  p.toolbar_location = None
  p.yaxis.visible = False
  tik_figures.append(p)

from bokeh.layouts import row
show(row(*tik_figures))

from bokeh.models import LinearColorMapper, BasicTicker, ColorBar, Title
## Figure plot
opt_codensity = codensity(np.mean(bw_bnds))
fp = figure_complex(
  S, pos=X, width=275, height=300, 
  color=opt_codensity, palette="inferno", 
  title="Noisy circle"
)
fp.match_aspect = True
fp.xaxis.visible = False
fp.yaxis.visible = False
fp.title.text_font_size = '20px'
# np.min(opt_codensity)
color_mapper = LinearColorMapper(palette="Inferno256", low=0.0, high=1.0)
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=0), location=(0,0), label_standoff=4)
# color_bar.title = Title(text="codensity", text_align='center', text_font_size='12px')
color_bar.title_standoff = 1
color_bar.width = 'auto'
color_bar.height = 15
color_bar.padding = 1
color_bar.title = "codensity"
color_bar.title_text_align = 'right'
color_bar.title_text_font_size = '18px'
fp.add_layout(color_bar, 'below')
fp.toolbar_location = None
fp.title.align = 'center'
fp.title.text_font_size = '20px'
show(fp)

# from pbsig.vis import figure_plain
# show(figure_plain(fp))

## Final plot vineyards 
from bokeh.layouts import row, column
a,b = 0.01, 0.015
vineyard_fig = figure_vineyard(dgms, p = 1, toolbar_location=None, height=300, width=300)
vineyard_fig.title = "Circle's H1 vineyard" 
vineyard_fig.line(x=[-10, a], y=[b, b], color='gray', line_dash='dashed', line_width=2) 
vineyard_fig.line(x=[a, a], y=[b, 10], color='gray', line_dash='solid', line_width=2) 
vineyard_fig.rect(x=a-5, y=b+5, width=10, height=10, fill_color='gray', fill_alpha=0.20, line_width=0.0)
vineyard_fig.scatter(x=a, y=b, color='red', size=8, legend_label="Sieve point")
vineyard_fig.legend.location = 'bottom_right'
vineyard_fig.legend.title_text_font_size = '16px'
vineyard_fig.xaxis.visible = False
vineyard_fig.yaxis.visible = False
vineyard_fig.add_layout(Label(x=a+0.001, y=b, text='(a,b)', text_color='red'))
vineyard_fig.title.text_font_size = '20px'
vineyard_fig.title.align = 'center'
color_mapper = LinearColorMapper(palette="Viridis256", low=0.0, high=1.0)
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=0), location=(0,0), label_standoff=4)
# color_bar.title = Title(text="codensity", text_align='center', text_font_size='12px')
color_bar.title_standoff = 1
color_bar.width = 'auto'
color_bar.height = 15
color_bar.padding = 1
color_bar.title = "bandwidth"
color_bar.title_text_font_size = '18px'
color_bar.title_text_align = 'center'
vineyard_fig.add_layout(color_bar, 'below')
show(vineyard_fig)


# Preview
show(row(fp, vineyard_fig, *tik_figures))

# Save
from bokeh.io import export_svg
for fig in [vineyard_fig, fp, *tik_figures]:
  fig.output_backend ='svg'
export_svg(row(fp, vineyard_fig, *tik_figures), filename="codensity_ex.svg")

# Get nice PNG 
from html2image import Html2Image
output_path = "/Users/mpiekenbrock/pbsig/applications"
hti = Html2Image(output_path=output_path)
hti.screenshot(
  other_file = output_path + '/codensity_ex.svg', 
  save_as = f'codensity_ex.png', 
  size = (1475, 300)
)

# inner latex width standard is 426 pt 
# page height is 674 pt 
ratio = (674 / 5.4) / 426


# from pbsig.linalg import tikhonov
# show(ri.figure_summary(tikhonov(eps=1.5, truncate="numrank"), post=scale_unit))







show(ri.figure_summary(func = spectral_rank, width=300, height=200, show_points=False))
show(ri.figure_summary(func = lambda x: x**2, post=scale_and_smooth, show_points=False))
show(ri.figure_summary(func = lambda x: np.abs(x)))
show(ri.figure_summary(func = lambda x: np.mean(x)))
show(ri.figure_summary(func = lambda x: x / (x + 1e-5), post=scale_and_smooth))
show(ri.figure_summary(func = tikhonov_truncated, post=scale_and_smooth))

show(ri.figure_summary(func = nuclear_interp(eps=1e-5)))
show(ri.figure_summary(func = heat(t=1000.4, complement=True)))











# %%
f = nth(ri.family, 59)
betti_query(S, f, matrix_func = spectral_rank, p = 1, i = 0.015, j = 0.02, normed = True, isometric = True, terms = True, w=ri.q_laplacian.sign_width)
433 - 157 - 297 + 22
# t0, t1, t2, t3 = betti_query(S, f, matrix_func = spectral_rank, p = 1, i = 0.015, j = 0.02, normed = True, isometric = True, terms = True, w=ri.q_laplacian.sign_width)

f = nth(ri.family, 60)
betti_query(S, f, matrix_func = spectral_rank, p = 1, i = 0.015, j = 0.02, normed = True, isometric = True, terms = True, w=ri.q_laplacian.sign_width)

t0, t1, t2, t3 = betti_query(S, f, matrix_func = lambda x: x, p = 1, i = 0.015, j = 0.02, normed = True, isometric = True, terms = True, w=ri.q_laplacian.sign_width)

t1
spectral_rank(t3)


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
# f1 = nth(ri.family, 43)
# f2 = nth(ri.family, 44)

# # %% 
# t1_a, t1_b, t1_c, t1_d = betti_query(S, f=f1, p = 1, matrix_func = lambda x: x, i = 0.015, j = 0.02, w=w, terms = True, form='array', normed=False, isometric=False)
# t2_a, t2_b, t2_c, t2_d = betti_query(S, f=f2, p = 1, matrix_func = lambda x: x, i = 0.015, j = 0.02, w=w, terms = True, form='array', normed=False, isometric=False)

# L = WeightedLaplacian(S, p = 1, w=w, normed=True, form='array')
# L.reweight()
# # np.array(L.op.fpl)
# # np.array(L.op.fpr)
# # np.array(L.op.fq)
# PsdSolver()(L.operator())
# np.sum(t_a > 1e-6)
# np.max(np.abs(f1(S) - f2(S)))



# %% Plot 
from bokeh.layouts import row, column
vineyard_fig = figure_vineyard(dgms, p = 1, toolbar_location=None, height=250, weidth=250)
vineyard_fig.line(x=[-10, 0.015], y=[0.02, 0.02], color='gray', line_dash='dashed', line_width=2) 
vineyard_fig.line(x=[0.015, 0.015], y=[0.02, 10], color='gray', line_dash='solid', line_width=2) 
vineyard_fig.rect(x=0.015-5, y=0.02+5, width=10, height=10, fill_color='gray', fill_alpha=0.20, line_width=0.0)
vineyard_fig.scatter(x=0.015, y=0.02, color='red', size=8)
show(vineyard_fig)

# ri.figure_summary(lambda x: x**2, post = normalize, figure = p)
from bokeh.models.annotations.labels import Label
normalize = lambda x: np.squeeze((x - np.min(x))/(np.max(x) - np.min(x)))
p = figure(width=350, height=250, toolbar_location=None, title="Spectral rank relaxations")
p.line(timepoints, normalize(ri.summarize(spectral_rank)), color='black')
p.line(timepoints, normalize(ri.summarize(lambda x: x**2)), color='blue')
p.line(timepoints, normalize(ri.summarize(np.abs)),  color='red')
p.line(timepoints, normalize(ri.summarize(tikhonov_truncated)),  color='green')
p.xaxis.axis_label = "Bandwidth"
p.yaxis.axis_label = "Spectral rank (normalized)"



from bokeh.layouts import row
show(row(vineyard_fig, p))

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

