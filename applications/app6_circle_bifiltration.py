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

output_notebook()

# %% Point cloud
np.random.seed(1234)
X = noisy_circle(150, n_noise=35)
S = sx.delaunay_complex(X)
# S = rips_complex(X, radius = np.quantile(pdist(X), 0.15)/2, p = 2)
show(figure_complex(S, X))

# %% Calculate filter functions
normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
vertex_density = gaussian_kde(X.T).evaluate(X.T)
f1 = lower_star_filter(normalize(max(vertex_density) - vertex_density))
f2 = flag_filter(normalize(pdist(X)))

# %% Obtain bigraded betti numbers
from pbsig.rivet import figure_betti
BI = bigraded_betti(S, f1, f2, p=1, xbin=15, ybin=15) # input_file="bifilt.txt", output_file="betti_out.txt")
show(figure_betti(BI))

# # %%
# from pbsig.rivet import anchors
# A = anchors(BI)
# from shapely import box, GeometryCollection
# from shapely import intersection_all, union_all
# boxes = [box(xmin=x, xmax=x+x_step, ymin=y, ymax=y+y_step) for x, y in zip(xg[xi], yg[yi])]
# boxes_geometry = GeometryCollection(boxes)
# union_all(boxes)

# %% Show the persistence diagram for a specific combination to optimize for
offset, angle = 0.10, 13.5
Y = np.c_[f1(S), f2(S)]
YL = push_map(Y, a=np.tan(np.deg2rad(angle)), b=offset)

# %%
K_proj = filtration(zip(YL[:, 2], S), form="rank")
dgm = ph(K_proj, engine="cpp")
show(figure_dgm(dgm[1]))  # should have 0.545 lifetime h1

# %% Make the parameterized family
degrees = np.linspace(angle - 5, angle + 5, 50)
bf_projects = [fixed_filter(S, push_map(Y, a=np.tan(np.deg2rad(d)), b=offset)[:, 2]) for d in degrees]
fibered_filter = ParameterizedFilter(S, family=bf_projects, p=1)
ri = SpectralRankInvariant(S, family=fibered_filter, p=1)
dgms = ri.compute_dgms(S)
vineyard_fig = figure_vineyard([d for d in dgms if 1 in d.keys()], p=1)
show(vineyard_fig)

# %% Show vineyard + its sieve
box = [0.265, 0.275, 0.965, 0.975]
ri.sieve = np.array([box])
ri.sift(w=1.0)
show(ri.figure_sieve(figure=vineyard_fig, fill_alpha=0.15))

# %% Verify rank
num_rank_f = np.vectorize(lambda x: 1.0 if np.abs(x) > 1e-6 else 0.0)
show(ri.figure_summary(func=num_rank_f))

# %% Aggregate across logarithmically-spaced regularization points
# Note this approach is purely local, thus would reflect a function to take the gradient of
g = lambda eps: np.ravel(ri.summarize(np.vectorize(lambda x: x / (x + eps))))
L = np.hstack([g(eps)[:, np.newaxis] for eps in np.geomspace(1e-8, 1e-3, 32)])
p = figure(width=350, height=250)
p.step(degrees, np.round(g(1e-6)), color="red")
p.line(degrees, L.mean(axis=1))
# p.line(degrees, (1.0/np.sqrt(L.shape[0])) * ((1.0/np.sqrt(L.std(axis=1))) * L.mean(axis=1)), color='purple')
show(p)



# %% 
from pbsig.rivet import anchors 
A = anchors(BI)
S = np.vstack((
	np.c_[BI['0']['x'], BI['0']['y']], 
	np.c_[BI['1']['x'], BI['1']['y']]
))
G = np.unique(np.vstack((S, A)), axis=0)  # (x,y)	
L = np.c_[G[:,0], -G[:,1]]								# (c,d) where y = cx - d
dual_pts = np.array([(xc := (b-d)/(c-a), a*xc + b) for (a,b),(c,d) in combinations(L,2) if (c-a) > 0])
xlim = [0, dual_pts[:,1].max()]
ylim = [min(-A[:,1].max(), A[:,1].min()), dual_pts[:,1].max()]


from shapely import LineString

LineString



# %% TODO: these are kind of not useful
# Try to create smoothing splines and average them

# g = lambda eps: np.ravel(ri.summarize(np.vectorize(lambda x: x / (x + eps))))
# s1, s1_res, _, _ = splrep(degrees, g(1e-1), s=0, full_output=True)
# s2, s2_res, _, _ = splrep(degrees, g(1e-2), s=0, full_output=True)
# s3, s3_res, _, _ = splrep(degrees, g(1e-3), s=0, full_output=True)
# c_avg = np.mean(np.vstack((s1[1], s2[1], s3[1])), axis=0)
# s_avg = BSpline(t=degrees, c=c_avg, k=3)

# d_points = np.linspace(degrees[0], degrees[-1], 250)
# p = figure(width=350, height=250)
# p.line(d_points, BSpline(*s1)(d_points), color="blue")
# p.line(d_points, BSpline(*s2)(d_points), color="red")
# p.line(d_points, BSpline(*s3)(d_points), color="green")
# p.line(d_points, s_avg(d_points), color="purple")
# show(p)
