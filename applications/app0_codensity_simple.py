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
np.random.seed(1235)
X = noisy_circle(200, n_noise=10, perturb=0.10) ## 80, 30, 0.15 
S = delaunay_complex(X)

# %% 
show(figure_complex(S, pos=X, width=300, height=300))

# %% Create filtering function
from scipy.stats import gaussian_kde
def codensity(bw: float):
  x_density = gaussian_kde(X.T, bw_method=bw).evaluate(X.T)
  x_codensity = max(x_density) - x_density
  return x_codensity

# %% Evaluate H1-persistence across a 1-parameter family
from splex import lower_star_filter
from pbsig.interpolate import ParameterizedFilter
bw_scott = gaussian_kde(X.T, bw_method='scott').factor
bw_bnds = (bw_scott*0.10, bw_scott*2)
timepoints = np.linspace(*bw_bnds, 100) # np.geomspace(*bw_bnds, 100)
codensity_family = [lower_star_filter(codensity(bw)/np.sum(codensity(bw))) for bw in timepoints] # note the normalization 
P = ParameterizedFilter(S, family = codensity_family, p = 1)
# P.interpolate(interval=(0, 1.8))

# %% 
from pbsig.betti import SpectralRankInvariant
ri = SpectralRankInvariant(S, family = codensity_family, p = 1)
# spri.randomize_sieve()
# show(spri.figure_sieve())

# %% Compute all the diagrams in the family
from pbsig.vis import figure_vineyard
dgms = ri.compute_dgms(S)
vineyard_fig = figure_vineyard(dgms, p = 1)

# %% Basically anything in [0, 0.01] x [0.02, 0.03] works
ri.sieve = np.array([[0.0, 0.01, 0.02, 0.03]])
show(ri.figure_sieve(figure=vineyard_fig, fill_alpha=0.15))

# %% Look at the numerical rank at a fixed threshold (that tends to work)
ri.sift(w=1.0) # w is a smoothing parameter
num_rank_f = np.vectorize(lambda x: 1.0 if np.abs(x) > 1e-6 else 0.0)
show(ri.figure_summary(func = num_rank_f))
show(ri.figure_summary(func = np.sum))

# %% Try multiple summary functions 
rank_relax_f = np.vectorize(lambda x: x / (x + 0.01))

show(ri.figure_summary(func = rank_relax_f))


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



# %% Check Parameterized lapalacina works 
# from pbsig.linalg import ParameterizedLaplacian
# P_laplacian = ParameterizedLaplacian(S, family = P, p = 0)

# [L for L in P_laplacian]



# %% Optimize via e.g. ISTA 

