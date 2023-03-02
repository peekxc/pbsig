# ---
# title: Circle optimization of varying density
# format: html
# editor:
#   render-on-save: true
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pbsig
#     language: python
#     name: python3
# ---

# %% Dependencies
import numpy as np
from splex import *
from pbsig import * 
from pbsig.linalg import * 
from pbsig.vis import *
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
output_notebook(verbose=False)


# %% Generate a noisy circle
from pbsig.datasets import noisy_circle
np.random.seed(1234)
X = noisy_circle(80, n_noise=30, perturb=0.15)
S = delaunay_complex(X)
# r = enclosing_radius(X)
# K = rips_filtration(X, r, p=2)

p = figure(width=300, height=300, match_aspect=True)
p.scatter(*X.T, color="red", size=10)
q = figure_complex(S, pos=X)
show(row(p, q))

# %% Color points by density + diameter
from splex.geometry import delaunay_complex
diam_f = flag_weight(pdist(X))
kde = gaussian_kde(X.T)
x_density = kde.evaluate(X.T)
x_codensity = max(x_density) - x_density
s_diam = np.array([diam_f(s) for s in faces(S)])
s_codensity = np.array([max(x_codensity[s]) for s in faces(S)])

p = figure(width=300, height=300, match_aspect=True,  title="Scatter by density")
p.scatter(*X.T, color=bin_color(x_density), size=12)
q = figure_complex(S, pos=X, color=x_codensity, title="Complex by codensity", simplex_kwargs={0: {'size': 10}})
r = figure_complex(S, pos=X, color=s_diam, title="Complex by diameter", simplex_kwargs={0: {'size': 10}})
show(row(p, q, r))

# %% Change density parameter 
alpha_family = np.linspace(kde.factor*0.10, kde.factor*1.5, 6)
figures = []
for a in alpha_family: 
  p = figure(width=200, height=200, title=f"Density: {a:.2f}")
  x_density = gaussian_kde(X.T, bw_method=a).evaluate(X.T)
  x_density /= max(x_density)
  p.scatter(*X.T, color=bin_color(x_density, lb=0, ub=1), size=8) # TODO: fix bounds?
  figures.append(p)
show(row(figures))


# %% Show the persistence diagrams
kw = dict(width=200, height=200)
cpx_figures, dgm_figures = [], []
for a in alpha_family: 
  x_density = gaussian_kde(X.T, bw_method=a).evaluate(X.T)
  x_density /= max(x_density)
  x_codensity = max(x_density) - x_density
  s_codensity = np.array([max(x_codensity[s]) for s in faces(S)])
  K = filtration(zip(s_codensity, faces(S)))
  dgm = ph(K, engine="dionysus")[1]
  dgm_figures.append(figure_dgm(dgm, x_range = (0, 1), y_range = (0, 1), **kw))
  cpx_figures.append(figure_complex(S, pos=X, color=s_codensity, title=f"alpha: {a:.2f}", simplex_kwargs={0: {'size': 10}}, **kw))
show(column(row(cpx_figures), row(dgm_figures)))

# %% Compute the the vineyards 
alpha_family = np.linspace(kde.factor*0.10, kde.factor*1.5, 100)
dgms = []
for a in alpha_family:
  x_density = gaussian_kde(X.T, bw_method=a).evaluate(X.T)
  x_density /= max(x_density)
  x_codensity = max(x_density) - x_density
  s_codensity = np.array([max(x_codensity[s]) for s in faces(S)])
  K = filtration(zip(s_codensity, faces(S)))
  dgms.append(ph(K, engine="dionysus")[1])
family_dgms = np.vstack([np.array((np.repeat(i, d.shape[0]), d['birth'], d['death'])).T for i,d in enumerate(dgms)])

# %% Plot the vineyards
p = figure_dgm(x_range = (0, 1), y_range = (0, 1))
p.scatter(family_dgms[:,1], family_dgms[:,2], color=bin_color(family_dgms[:,0]))
show(p)

# %% Start by plotting the continuously varying weight function
import functools
codensity_family = []
for a in alpha_family:
  x_density = gaussian_kde(X.T, bw_method=a).evaluate(X.T)
  x_density /= max(x_density)
  x_codensity = max(x_density) - x_density
  codensity_family.append(functools.partial(lambda s, c: max(c[s]), c=x_codensity))

p = figure(width=400, height=250)
for s in faces(S):
  s_weights = np.array([[a, f(s)] for a, f in zip(alpha_family, codensity_family)])
  p.line(s_weights[:,0], s_weights[:,1])
show(p)

# %% 
from pbsig.betti import MuFamily
R = (0.2, 0.4, 0.6, 0.8)
mu_f = MuFamily(S, codensity_family, p = 1)
mu_f.precompute(R, w=0.0, normed=True)

# %% Verify multiplicity matches with persistence diagram visually
p = figure_dgm(x_range=(0, 1), y_range=(0, 1), width=250, height=250)
p.scatter(family_dgms[:,1], family_dgms[:,2], color=bin_color(family_dgms[:,0]))
r_width, r_height = R[1]-R[0], R[3]-R[2]
p.rect(R[0]+r_width/2, R[2]+r_height/2, r_width, r_height, alpha=1.0, fill_alpha=0.0)
q = figure(width=300, height=250, title="Multiplicity")
q.step(alpha_family, mu_f())
show(row(p,q))

# %% Look at the other relaxations
from pbsig.linalg import *
mu_rk = mu_f(smoothing=None, terms=False)
mu_nn = mu_f(smoothing=False, terms=False)
mu_sa = mu_f(smoothing=sgn_approx(eps=1e-1, p=1.0), terms=False)

p = figure(width=300, height=250, title="Multiplicity")
p.step(alpha_family, mu_rk, line_color="black")
p.line(alpha_family, mu_nn, line_color="red")
p.line(alpha_family, mu_sa, line_color="blue")
show(p)

# %% Constitutive terms 
mu_rk = mu_f(smoothing=None, terms=True).T
mu_nn = mu_f(smoothing=False, terms=True).T
mu_sa = mu_f(smoothing=sgn_approx(eps=1e-1, p=1.0), terms=True).T

figures = []
for i in range(4):
  p = figure(width=300, height=250, title="Multiplicity")
  p.step(alpha_family, mu_rk[:,i], line_color="black")
  p.line(alpha_family, mu_nn[:,i], line_color="red")
  p.line(alpha_family, mu_sa[:,i], line_color="blue")
  figures.append(p)
show(row(figures))

# %% Gradient calculation 
import numdifftools as nd

# Start with a continuously parameterized family
import functools
def codensity(alpha: float):
  x_density = gaussian_kde(X.T, bw_method=alpha).evaluate(X.T)
  x_density /= max(x_density)
  x_codensity = max(x_density) - x_density
  return functools.partial(lambda s, c: max(c[s]), c=x_codensity)

from pbsig.betti import mu_query
def codensity_mu(alpha: float):
  return mu_query(S, R=R, f=codensity(alpha), p=1, w=0.40, normed=True)(smoothing=sgn_approx(eps=1e-2, p=2.0), terms=False)


codensity_d(0.05)




## Takes 10 seconds to evaluate a single derivative! 
# from scipy.sparse.linalg import eigsh
# from splex.sparse import _boundary
# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(codensity_mu)
# profile.add_function(mu_query)
# profile.add_function(_boundary)
# profile.add_function(polymorphic_psd_solver)
# profile.add_function(eigsh)
# profile.enable_by_count()
# codensity_d = nd.Derivative(codensity_mu, n=1, order=1)
# codensity_d(0.05)
# profile.print_stats(output_unit=1e-3)
























# # %% Start by varying w
# # mu_f.precompute(R, w=0.40, normed=True)
# # mu_rk = mu_f(smoothing=None, terms=False)
# # mu_nn = mu_f(smoothing=False, terms=False)
# # mu_sa = mu_f(smoothing=sgn_approx(eps=1e-3, p=2.0), terms=False)
# # p = figure(width=300, height=250, title="Multiplicity")
# # p.step(alpha_family, mu_rk, line_color="black")
# # p.line(alpha_family, mu_nn, line_color="red")
# # p.line(alpha_family, mu_sa, line_color="blue")
# # show(p)

# # %% Check out the Moreau envelope 


# # %% [markdown]
# # The goal is to create a 1-parameter family of weight functions $w_\alpha : K \to \mathbb{R}$ whose corresponding 
# # vineyard $\{ \mathrm{dgm}(\alpha) \}_{\alpha \in \mathcal{A}}$ maps out interesting topological behavior. 
# # In this case, we use combination of the Rips and (co)density functions: 
# # $$ 
# # \begin{align*}
# #   w  & : & \mathbb{R} & \to & \,  \mathbb{R}^{\lvert K \rvert}  \\
# #     & & \alpha & \mapsto &  \{ \alpha \cdot \mathrm{codensity}(\sigma) + \mathrm{diam}(\sigma) \}
# # \end{align*}
# # $$
# # For any $\alpha \geq 0$. 

# x_codensity = max(x_density) - x_density

# # from pbsig.vis import figure_complex
# # alpha_family = np.linspace(0, 50, 10)
# # figures, dgm_figures = [], []
# # for a in alpha_family: 
# #   fv = a * s_codensity + s_diam
# #   p = figure_complex(S, pos=X, color=fv, title=f"alpha: {a:.2f}", simplex_kwargs={0: {'size': 10}}, width=200, height=200)
# #   K = filtration(zip(fv, faces(S)))
# #   dgm_figures.append(figure_dgm(ph(K, engine="dionysus")[1], width=200, height=200))
# #   figures.append(p)
# # show(column(row(figures), row(dgm_figures)))


# # %%
# from pbsig.persistence import ph
# radius = 1.2*r
# KC = rips_filtration(circle, radius, p=2)
# KN = rips_filtration(X, radius, p=2)

# dgm_C = ph(KC, engine="dionysus")[1]
# dgm_N = ph(KN, engine="dionysus")[1]
# show(row(figure_dgm(dgm_C), figure_dgm(dgm_N)))

# # %% [markdown]
# # ## The parameterized family
# #
# #

# # %%
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import gaussian_kde
# kde = gaussian_kde(X.T)
# x_density = kde.evaluate(X.T)
# x_dist = pdist(X)
# K = rips_filtration(x_dist, radius=enclosing_radius(x_dist), p=2)

# simplices = list(K.values())
# s_diameter = np.array(list(K.keys()))
# s_density = np.array([min(x_density[s]) for s in faces(K)])
# s_codensity = max(s_density) - s_density

# from pbsig.vis import figure_complex
# p = figure_complex(K, pos=X, color=x_density)
# p.scatter(*X.T, fill_color="black", fill_alpha=0.0, line_color="black", line_width=1.5)
# show(p)

# p = figure(width=350, height=300, title="")
# p.scatter(*X.T, color=bin_color(s_codensity), size=5)
# p.xaxis.axis_label = "codensity"
# p.yaxis.axis_label = "rips diameter"
# show(p)



# # %%
# ## Choose a line in the plane to project the bilfitration onto
# def push_map(X, a, b):
#   n = X.shape[0] 
#   out = np.zeros((n, 3))
#   for i, (x,y) in enumerate(X):
#     tmp = ((y-b)/a, y) if (x*a + b) < y else (x, a*x + b)
#     dist_to_end = np.sqrt(tmp[0]**2 + (b-tmp[1])**2)
#     out[i,:] = np.append(tmp, dist_to_end)
#   return(out)
# Y = np.c_[s_codensity, s_diameter]
# YL = push_map(Y, a=15.0, b=0.0)

# p = figure(width=450, height=350, title="Bifiltration")
# p.scatter(Y[:,0], Y[:,1], color='blue', size=1.5)
# p.xaxis.axis_label = "codensity"
# p.yaxis.axis_label = "rips diameter"
# p.scatter(YL[:,0], YL[:,1], color=bin_color(YL[:,2], "turbo"))
# show(p)

# # %%
# from bokeh.models import Slope
# p = figure(width=450, height=350, title="Bifiltration")
# p.scatter(Y[:,0], Y[:,1], color='blue', size=1.5)
# p.xaxis.axis_label = "codensity"
# p.yaxis.axis_label = "rips diameter"

# lb = Slope(gradient=1.0, y_intercept=0,line_color='red', line_dash='dashed', line_width=1.5)
# ub = Slope(gradient=15.0, y_intercept=0,line_color='red', line_dash='dashed', line_width=1.5)
# YL = push_map(Y, a=7.5, b=0.0)
# p.scatter(YL[:,0], YL[:,1], color=bin_color(YL[:,2], "turbo"))
# p.add_layout(lb)
# p.add_layout(ub)
# show(p)

# # %%
# slope_family = np.linspace(1.0, 15.0, 30)
# push_map_dgms = []
# for m in slope_family:
#   YL = push_map(Y, a=m, b=0.0)
#   K = filtration(zip(YL[:,2], simplices))
#   push_map_dgms.append(ph(K, engine="dionysus"))

# # %%
# p = figure_dgm(title="Pushmapped family of filtrations")
# sc = bin_color(np.arange(len(push_map_dgms)), "turbo")
# for i, dgm in enumerate(push_map_dgms):
#   nf = len(dgm[1]['death'])
#   pt_col = np.reshape(np.repeat(sc[i,:], nf), (4,nf)).T
#   p.scatter(dgm[1]['birth'], dgm[1]['death'], color=pt_col)
# max_death = max([max(d[1]['death']) for d in push_map_dgms])
# p.x_range = Range1d(0, max_death*1.05)
# p.y_range = Range1d(0, max_death*1.05)
# show(p)

# # %%
# from pbsig.betti import MuFamily

# ## Choose a rectangle 
# S = simplicial_complex(simplices)
# R = (0.2, 0.4, 0.6, 0.8)
# def push_f(a,b):
#   def _push(s) -> float:
#     return max(push_map(X[s,:], a, b)[:,2])
#   return _push

# F = [push_f(a, 0.0) for a in slope_family]
# mu_f = MuFamily(S, family=F, p=1, form="array")
# mu_f.precompute(R=R, progress=True)

# # %%
# mu_f(terms=True)

# # %%
# from pbsig.betti import mu_query
# mu_q = mu_query(S, p=1, R=R, f=push_f(7.5, 0.0))
# mu_q(terms=True)

# # %% [markdown]
# # An alternative is to use the Fermat distance for varying parameter $p \geq 1$, though this needs some normalization (probably) to understand the choice of $R$. 

# # %%
# ## Intrinsic persistence
# from scipy.spatial.distance import pdist, squareform
# from scipy.sparse.csgraph import floyd_warshall
# intrinsic_dgms = {}
# for p in progressbar(np.linspace(1.0, 3.0, 10)):
#   d = floyd_warshall(squareform(pdist(X)**p))
#   d = d / max(np.ravel(d))
#   K = rips_filtration(d, radius=enclosing_radius(d), p=2)
#   dgm = ph(K, engine="dionysus")[1]
#   intrinsic_dgms[p] = dgm

# # %%
# figures = []
# for p, dgm in intrinsic_dgms.items():
#   q = figure_dgm(width=200, height=200, title=f"Intrinsic pers (p={p:.2f})")
#   q.scatter(dgm['birth'], dgm['death'])
#   # q.x_range = Range1d(0, max(dgm['death'])*1.05)
#   # q.y_range = Range1d(0, max(dgm['death'])*1.05)
#   q.x_range = Range1d(0,2.0)
#   q.y_range = Range1d(0,2.0)
#   figures.append(q)
# show(column(row(figures[:5]), row(figures[5:])))

# # %%
# from pbsig.betti import mu_query
# R = (0.5, 1.0, 1.5, 2.0)
# S = simplicial_complex(faces(SimplexTree([np.arange(8)]), 2))
# Q = [mu_query(S, R=R, f=flag_weight(alpha*circle), p=1, normed=False) for alpha in alpha_family]

# mult_H1 = [mu() for mu in Q]
# p = figure(
#   width=350, height=250, 
#   title=f"Circle multiplicities for R={R}", x_axis_label="alpha (scaling factor)", y_axis_label="multiplicity"
# )
# p.step(alpha_family, np.array(mult_H1, dtype=int))
# p.yaxis.minor_tick_line_alpha = 0

# q = figure_dgm(width=250, height=250)
# q.scatter(vine[:,0],vine[:,1], color=bin_color(alpha_family, "turbo"))
# q.x_range = Range1d(0, max(vine[:,1])*1.05)
# q.y_range = Range1d(0, max(vine[:,1])*1.05)
# r_width = R[1]-R[0]
# r_height = R[3]-R[2]
# q.rect(R[0]+r_width/2, R[2]+r_height/2, r_width, r_height, alpha=1.0, fill_alpha=0.0)


# show(row(p, q))

# # %% [markdown]
# # Equivalently, we can amortize the cost of creating so many matrices by re-using the results from previous computations.
# #

# # %%
# from pbsig.betti import MuFamily
# F = [flag_weight(alpha*circle) for alpha in alpha_family]
# mu_f = MuFamily(S, family=F, p=1, form="array")
# mu_f.precompute(R=R, progress=True)

# # %% [markdown]
# # Let's look at the constitutive terms that make up this multiplicity queries
# #

# # %%
# p = figure(
#   width=450, height=250, 
#   title=f"Circle multiplicities for R={R}", 
#   x_axis_label="alpha (scaling factor)", 
#   y_axis_label="Constititive ranks"
# )
# mu_terms = mu_f(smoothing=None, terms=True).T
# p.step(alpha_family, mu_terms[:,0], line_color="red")
# p.step(alpha_family, -mu_terms[:,1], line_color="green")
# p.step(alpha_family, -mu_terms[:,2], line_color="blue")
# p.step(alpha_family, mu_terms[:,3], line_color="orange")
# p.step(alpha_family, mu_f(smoothing=None), line_color="black")
# p.yaxis.minor_tick_line_alpha = 0
# show(p)

# # %% [markdown]
# # Note that there may be regions $\bar{\mathcal{A}} \subset \mathbb{R}$ wherein all four terms in $\mu_R(\alpha)$ are $0$. Thus, even if one equipped some differentiable structure to the rank function, the gradient calculation for $\alpha \in \bar{\mathcal{A}}$ would still be $0$. In other words, there is some interval $\mathcal{A} \subset \mathbb{R}$ wherein an theoretical optimization is feasible, and outside of this region there is no hope at optimization. The bounds of $\mathcal{A}$ are given below:

# # %%
# feasible = np.flatnonzero(np.diff((abs(mu_terms).sum(axis=1) != 0).astype(int)))
# f"min alpha: {alpha_family[feasible[0]]:4f}, max alpha: {alpha_family[feasible[1]]:4f}"

# # %% [markdown]
# #
# # Now, let's look at a continuous relaxation of both the multiplicity function and its constitituive terms.
# #

# # %%
# ## constituive terms
# figures = [[] for i in range(4)]
# for w, normed in [(0.0,False), (0.0,True), (0.30,False), (0.30,True)]:
#   mu_f.precompute(R=R, w=w, normed=normed, biased=True, progress=False)
#   mu_terms = mu_f(smoothing=None, terms=True).T
#   mu_terms_nuclear = mu_f(smoothing=False, terms=True).T
#   mu_terms_sgn_approx = mu_f(smoothing=sgn_approx(eps=1e-1, p=1.5), terms=True).T
#   for i in range(4):
#     p = figure(
#       width=200, height=200, 
#       title=f"w:{w}, normed:{normed}", x_axis_label="alpha (scaling factor)", y_axis_label="multiplicity",
#       tools="",
#     )
#     p.toolbar.logo = None
#     p.yaxis.minor_tick_line_alpha = 0
#     p.line(alpha_family, mu_terms_nuclear[:,i], line_color = "orange", line_width=2.0)
#     p.line(alpha_family, mu_terms_sgn_approx[:,i], line_color = "blue",  line_width=1.5)
#     p.step(alpha_family, mu_terms[:,i], line_color = "black", line_width=1.0)
#     figures[i].append(p)

# show(row([column(f) for f in figures]))

# # %% [markdown]
# # **TAKEAWAY**: If $w > 0$, you need degree normalization to stabilize the spectrum. Though the nuclear norm is at times quite similar to the rank, it can also differ greatly.
# #
# # The weighted combinatorial laplacian is unstable whenever $w > 0$, due to the fact that $1/\epsilon$ can produce arbitrarily large values as $\epsilon \to 0^+$. This is not a problem for the normalized laplacian, as the spectrum is always bounded in the range $[0, p+2]$.
# #
# # ------------------------------------------------------------------------
# #
# # From now on, let's only consider the normalized weighted combinatorial laplacian. Let's look at the effect of w
# #

# # %%
# W = [0.0, 0.30, 0.60]
# fig_kwargs = dict(width=200, height=150)

# ## Try varying epsilon in [1e-1, 100] to see interpolation of behavior
# figures = [[] for i in range(len(W))]
# for i,w in enumerate(W):
#   mu_f.precompute(R=R, w=w, normed=True, progress=False)
#   headers = ["rank", "sgn_approx", "nuclear"]
#   for j, (ef, title) in enumerate(zip([None, sgn_approx(eps=1e-1, p=1.2), False], headers)):
#     mu_terms = mu_f(smoothing=ef, terms=True).T
#     p = figure(**(fig_kwargs | dict(title=f"w:{w}, f:{title}")), tools="")
#     p.toolbar.logo = None
#     p.line(alpha_family, mu_terms[:,0], line_color="red")
#     p.line(alpha_family, -mu_terms[:,1], line_color="green")
#     p.line(alpha_family, -mu_terms[:,2], line_color="blue")
#     p.line(alpha_family, mu_terms[:,3], line_color="orange")
#     p.line(alpha_family, mu_terms.sum(axis=1), line_color="black")
#     p.yaxis.minor_tick_line_alpha = 0
#     figures[i].append(p)

# show(column([row(f) for f in figures]))

# # %% [markdown]
# # On the positive, changing `w` smoothes the objective nicely to some extent and it also expands the interval $\mu(\alpha) \neq 0$ within the feasible set. 
# #
# # On the negative, the nuclear norm causes a spurious maximizer and is $0$ in the region we're trying to maximize. Moreover, varying `w` has seemingly neglible impact on the nuclear norm.

# # %%
# # mu_f.precompute(R=R, w=0.30, normed=True, progress=False)
# # for j,ef in enumerate([None, sgn_approx(eps=1e-1, p=1.2), False]):
# #   mu_terms = mu_f(smoothing=ef, terms=True).T
# # feasible = np.flatnonzero(np.diff((abs(mu_terms).sum(axis=1) != 0).astype(int)))
# # f"min alpha: {alpha_family[feasible[0]]:4f}, max alpha: {alpha_family[feasible[1]]:4f}"

# # %% [markdown]
# # Let's zoom in on the sgn approximation

# # %%
# mu_f.precompute(R=R, w=0.30, normed=True, biased=True, progress=False)
# ef = sgn_approx(eps=1e-1, p=1.2)
# mu_terms = mu_f(smoothing=ef, terms=True).T
# p = figure(**(fig_kwargs | dict(title=f"w:{w}, f:sgn_approx", width=450, height=250)))
# p.toolbar.logo = None
# p.line(alpha_family, mu_terms[:,0], line_color="red")
# p.line(alpha_family, -mu_terms[:,1], line_color="green")
# p.line(alpha_family, -mu_terms[:,2], line_color="blue")
# p.line(alpha_family, mu_terms[:,3], line_color="orange")
# p.line(alpha_family, mu_terms.sum(axis=1), line_color="black")
# p.yaxis.minor_tick_line_alpha = 0

# show(p)

# # %% [markdown]
# # The cancellations of the green/orange (2nd/4rth terms) and red/blue (1st/3rd terms) render ~1/3 of the feasible region useless _before_ the maximizer; the green/red and blue/orange also ~1/3 useless _after_. 
# #
# # Questions that remain: 
# #   1. Can the region where $\hat{\mu_R} \neq 0$ be expanded?
# #   2. Can the objective be further smoothed _while retaining maximizers_? 
# # ---

# # %% [markdown]
# # A common approach for retaining critical points while smoothing is the Moreau envelope of a function $f$. It is defined as attained infimum of the proximal operator of $\mathrm{prox}_{f}$. It is defined as:
# #
# # $$ \mathrm{prox}_{\lVert \cdot \rVert_\ast}(X) = \mathrm{inf}_{Y \in \mathbb{R^{m\times m}}} \big ( \lVert Y \rVert_\ast + \frac{1}{2t}\lVert X - Y\rVert_F^2 \big ) $$
# #
# # Note that the quadratic term becomes blows up as $t \to 0^+$, implying smaller values of $t$ makes the corresponding Moreau envelope closer to $\lVert \cdot \rVert_\ast$. In effect,  smaller values $t$ act like an increasingly tighter neighborhood restriction in $\mathbb{R}^{m \times m}$.

# # %%
# w = 0.30
# mu_f.precompute(R=R, w=w, normed=True, progress=True) # use nuclear norm

# # %%
# from pbsig.linalg import *
# from pbsig.utility import *

# mu_terms = mu_f(smoothing=True, terms=True).T
# mu_mats = [mu_query_mat(S=S, R=R, f=f, p=1, w=w, form = 'array', normed=True) 
#            for f in mu_f.family]
# me = lambda M, t: prox_nuclear(M, t)[1]

# figures = []
# for t in [1e-1, 0.5, 1.0, 10.0]:
#   mu_moreau = [[] for i in range(4)]
#   for M1, M2, M3, M4 in mu_mats:
#     mu_moreau[0].append(me(M1, t))
#     mu_moreau[1].append(-me(M2, t))
#     mu_moreau[2].append(-me(M3, t))
#     mu_moreau[3].append(me(M4, t))
#   mu_moreau = np.array(mu_moreau).T
#   p = figure(**(fig_kwargs | dict(title=f"w:{w}, f:nuclear, t:{t}", tools="")))
#   p.toolbar.logo = None
#   p.line(alpha_family, mu_terms.sum(axis=1), line_color="black")
#   p.line(alpha_family, mu_moreau.sum(axis=1), line_color="black", line_dash="dotted", line_width=1.5)
#   p.yaxis.minor_tick_line_alpha = 0
#   figures.append(p)

# show(row(figures))


# # %% [markdown]
# # Even if the function $f$ is not convex lower semi-continuous, the minimizers of the (smooth) Moreau regularization can match the minimizers of the original function with the right parameters. Letting $t \to 0_+$ ensures that. Since the $\mathrm{sign}$ approximation is arbitrarily close to the rank function, why not take the Moreau envelope of that? 

# # %%
# def moreau_loss(x: ArrayLike, sf: Callable, t: float = 1.0):
#   """
#   Parameters:
#     x:  ndarray, spmatrix, or LinearOperator
#     sf: element-wise function to apply to the eigenvalues 
#     t:  proximal scaling operator 
#   """
#   from scipy.sparse import spmatrix
#   from scipy.optimize import minimize
#   if x.ndim == 0: return 0
#   if isinstance(x, np.ndarray) and all(np.ravel(x == 0)):
#     return 0
#   elif isinstance(x, spmatrix) and (len(x.data) == 0 or x.nnz == 0):
#     return 0
#   #x_ew = np.linalg.eigvalsh(x)
#   x_ew = eigvalsh_solver(x)(x)
#   def sgn_approx_cost(ew_hat: ArrayLike, t: float):
#     return sum(sf(ew_hat)) + (1/(t*2)) * np.linalg.norm(sf(ew_hat) - sf(x_ew))**2
#   w = minimize(sgn_approx_cost, x0=x_ew, args=(t), tol=1e-15, method="Powell")
#   if w.status != 0:
#     import warnings
#     warnings.warn("Did not converge to sign vector prox")
#   ew = w.x if w.status == 0 else x_ew
#   return sgn_approx_cost(ew, t)

# w = 0.30   # hyper-parameter for R
# eps = 1e-1 # hyper-parameter for sgn
# sf = sgn_approx(eps=eps, p=1.5)
# mu_terms = mu_f(smoothing=sf, terms=True).T
# mu_mats = [mu_query_mat(S=S, R=R, f=f, p=1, w=w, biased=True, form = 'array', normed=True) 
#            for f in mu_f.family]

# figures = []
# for t in [1e-2, 1e-1, 1.0, 10.0]:
#   mu_moreau = [[] for i in range(4)]
#   for M1, M2, M3, M4 in mu_mats:
#     mu_moreau[0].append(moreau_loss(M1, sf, t=t))
#     mu_moreau[1].append(-moreau_loss(M2, sf, t=t))
#     mu_moreau[2].append(-moreau_loss(M3, sf, t=t))
#     mu_moreau[3].append(moreau_loss(M4, sf, t=t))
#   mu_moreau = np.array(mu_moreau).T
#   p = figure(**(fig_kwargs | dict(title=f"w:{w}, f:sgn_approx, t:{t}", tools="")))
#   p.toolbar.logo = None
#   p.line(alpha_family, mu_terms.sum(axis=1), line_color="black")
#   p.line(alpha_family, mu_moreau.sum(axis=1), line_color="black", line_dash="dotted", line_width=1.5)
#   p.yaxis.minor_tick_line_alpha = 0
#   figures.append(p)

# show(row(figures))

# # %% [markdown]
# # Very interesting! The symmetrized version of `w` sort of uniformly smoothes out the objective! It's true that the constititive matrix components will not have the correct rank---but, they will as $w \to 0^+$. This motivates the strategy: start with the symmetric setting of $w > 0$ to smooth out the objective, and slowly decrease it (via temperature/cooling parameter?) to get closer to the rank.
# #
# # Moreover, the symmetrized version _should_ increase the domain where $\mu \neq 0$. 
# #
# # ---

# # %% [markdown]
# # # Gradient ascent 
# #
# # For this we can just use the proximal gradient algorithm. 
# #

# # %%
# import autograd as ad
# eps, p = 1e-1, 1.5
# phi = lambda x: ad.sign(x) * (ad.abs(x)**p / (ad.abs(x)**p + eps**p))
# grad_phi = ad.grad(phi)

# ## need gradient of proximal 
# ## and gradient via finite diffference of laplacians 


# # %%
# ## Empirically get at the proximal operator of our function 
# from scipy.optimize import minimize_scalar

# ## Recover the proximal operator for the absolute value / sgn approx
# # f, fname = lambda x: np.abs(x), "abs value"
# f, fname = lambda x: abs(sgn_approx(eps=1.2, p=1.5)(x)), "sgn approx"

# def prox_f(x: float, t: float):
#   dual = lambda z, t: f(z) + 1/(2*t) * (x-z)**2
#   opt = minimize_scalar(dual, bracket=(x-10.0, x+10.0), args=(t))
#   if opt.success:
#     return opt.x, opt.fun
#   else: 
#     import warnings
#     warnings.warn("failed to minimize")
#     return x, 0.0

# x_dom = np.linspace(-10, 10, 1000)


# prox_figures, moreau_figures = [], []
# for t in [1e-12, 1e-1, 1.0, 5.0]:
#   prox_x = [prox_f(x, t)[0] for x in x_dom]
#   m_env = [prox_f(x, t)[1] for x in x_dom]
#   p = figure(width=200, height=200, title=f"Prox f: {fname}, t: {t}")
#   p.line(x_dom, prox_x)
#   m = figure(width=200, height=200, title=f"Moreau f: {fname}, t: {t}")
#   m.line(x_dom, m_env)
#   prox_figures.append(p)
#   moreau_figures.append(m)
# show(column(row(prox_figures), row(moreau_figures)))


# # %%
# # prox_eps = lambda x,t,eps: 0.5*(-t*x + np.sqrt(t**2 * x**2 + 4 * t * eps * (x + eps))) if x >= 0 else 0.5*(-t*x - np.sqrt(t**2 * x**2 + 4 * t * eps * (x - eps)))

# # wut = [np.sign(x)*prox_eps(np.abs(x), 0.5, 0.5) for x in np.linspace(-10,10,1000)]
# # p = figure()
# # p.line(np.linspace(-10,10,1000), wut)
# # show(p)

# # %%
# ## Testing the moreau envelope 
# from pbsig.linalg import prox_nuclear
# X = np.random.uniform(size=(10,10))
# X = X @ X.T

# from pbsig.linalg import soft_threshold
# t = 0.15
# ew, ev = np.linalg.eigh(X)
# assert np.isclose(sum(np.linalg.eigvalsh(ev @ np.diag(ew) @ ev.T)), sum(ew))
# assert np.isclose(sum(np.linalg.eigvalsh(ev @ np.diag(soft_threshold(ew, 0.15)) @ ev.T)), sum(soft_threshold(ew, 0.15)))
# P = ev @ np.diag(soft_threshold(ew, 0.15)) @ ev.T  #  proximal operator 
# assert np.isclose(sum(np.linalg.eigvalsh(P)), sum(soft_threshold(ew, 0.15)))
# me = sum(soft_threshold(ew, 0.15)) + (1/(2*0.15))*np.linalg.norm(X - P, 'fro')**2
# P, mf, _ = prox_nuclear(X, t=t) # 35.36999221118293
# assert np.isclose(me, mf)


# A = np.random.uniform(size=(10,10))
# A = A @ A.T
# print(np.linalg.norm(A - X, 'fro')**2)
# ew_x, U =np.linalg.eigh(X)
# ew_v, V =np.linalg.eigh(A)
# S = np.diag(ew_x)
# D = np.diag(ew_v)

# np.trace(S**2) + np.trace(D**2) - 2*np.trace(V.T @ U @ S @ U.T @ V @ D)
# np.trace(S**2) + np.trace(D**2) - 2*np.trace(S**2 @ U.T @ X @ D @ X.T @ U)

# sum((np.diag(A) - np.diag(X))**2)
# sum((np.diag(A)**2 - np.diag(X)**2))



# from pbsig.linalg import moreau
# sum(moreau(ew, t))

# # %%
# from pbsig.linalg import eigh_solver, eigvalsh_solver
# from pbsig.betti import mu_query_mat
# sf = sgn_approx(eps=1e-2, p=1.2)  
# LM = mu_query_mat(S, R=R, f=F[jj], p=1, form="array")
# Y = LM[3].todense()
# #ew, ev = eigh_solver(Y)(Y)

# y_shape = Y.shape
# y = np.ravel(Y)
# def moreau_cost(y_hat: ArrayLike, t: float = 1.0):
#   Y_hat = y_hat.reshape(y_shape)
#   ew = np.maximum(np.real(np.linalg.eigvals(Y_hat)), 0.0) # eiegnvalues can be negative, so we project onto the PSD cone!
#   ew_yhat = sum(sf(ew)) 
#   if t == 0.0: 
#     return ew_yhat
#   return ew_yhat + (1/(t*2))*np.linalg.norm(Y_hat - Y, "fro")**2
# # ev @ np.diag(sf(ew)) @ ev.T

# from scipy.optimize import minimize
# y_noise = y+np.random.uniform(size=len(y), low=0, high=0.01)
# w = minimize(moreau_cost, x0=y_noise, args=(0.01))
# Z = w.x.reshape(y_shape)
# print(f"Status: {w.status}, total error: {np.linalg.norm(Z - Y, 'fro')}, Num it: {w.nit}, Num evals: {w.nfev} \nMessage: {w.message}")

# ## Try a vector based optimization
# from scipy.optimize import minimize
# y_ew = np.linalg.eigvalsh(Y)
# def sgn_approx_cost(ew_hat: ArrayLike, t: float = 1.0):
#   return sum(sf(ew_hat)) + (1/(t*2)) * np.linalg.norm(sf(ew_hat) - sf(y_ew))**2
# y_ew_noise = y_ew + np.random.uniform(size=len(y_ew), low=0.0, high=0.50)
# w = minimize(sgn_approx_cost, x0=y_ew_noise, args=(0.5), tol=1e-15, method="Powell")
# print(f"Status: {w.status}, total error: {np.linalg.norm(y_ew - w.x)}, Num it: {w.nit}, Num evals: {w.nfev} \nMessage: {w.message}")



# eigvalsh_solver(Z)(Z)

# mu_query(S, R, f)
# solver = eigh_solver(x)
# ew, ev = solver(x)


# x0 

# # j = np.searchsorted(alpha_thresholds, 0.90)
# # mu = mu_query(S, R=np.append(R, 0.35), f=F[j], p=1, normed=True)
# # mu(smoothing=None, terms=False) # 1 
# # mu(smoothing=None, terms=True) # 8,  -8, -16,  17
# # mu(smoothing=sgn_approx(eps=1e-2, p=1.2), terms=False) # 1.0004954387928944
# # mu(smoothing=sgn_approx(eps=0.90, p=1.0), terms=False) # 0.5891569860030863
# # mu(smoothing=False, terms=True) #  8., -16., -16.,  24.

# # jj = np.searchsorted(alpha_thresholds, 1.15)
# # f=F[jj]
# # spectral_rank(EW[0])

# # mu = mu_query(S, R=np.append(R, 0.35), f=F[jj], p=1, normed=True)
# # mu(smoothing=False, terms=True)

# # %%
# mu_f.precompute(R=R, normed=False, progress=True)

# # %% [markdown]
# # Use discrete vineyards to get an idea of what the
# #
# # # st = SimplexTree(complete_graph(X.shape\[0\]))
# #
# # # st.expand(2)
# #
# # # S = st
# #
# # N, M = 20, 24 SW = sliding_window(sw_f, bounds=(0, 12*np.pi)) d, tau = sw_parameters(bounds=(0,12*np.pi), d=M, L=6) #S = delaunay_complex(F(n=N, d=M, tau=tau)) X = SW(n=N, d=M, tau=tau) \# r = enclosing_radius(X)\*0.60 \# S = rips_complex(X, r, 2) show(plot_complex(S, X\[:,:2\]))
# #
# # ## Plot
# #
# # scatters = \[\] for t in np.linspace(0.50*tau, 1.50*tau, 10): X_delay = SW(n=N, d=M, tau=t) p = figure(width=150, height=150, toolbar_location=None) p.scatter(*pca(X_delay).T) scatters.append(plot_complex(S, pos=pca(X_delay), width=125, height=125)) show(row(*scatters))
# #
# # from pbsig.persistence import ph from pbsig.vis import plot_dgm K = filtration(S, f=flag_weight(X)) dgm = ph(K, engine="dionysus") plot_dgm(dgm\[1\])
# #
# # from pbsig.betti import MuSignature, mu_query from pbsig.linalg import \* R = np.array(\[4, 4.5, 6.5, 7.5\]) T_dom = np.append(np.linspace(0.87*tau, tau, 150, endpoint=False), np.linspace(tau, tau*1.12, 150, endpoint=False)) t_family = \[flag_weight(SW(n=N, d=M, tau=t)) for t in T_dom\]
# #
# # MU_f = mu_query(S, R=R, f=flag_weight(SW(n=N, d=M, tau=tau)), p=1, form="array")
# #
# # Generate a noisy circle
# #

# # %%
# np.random.seed(1234)
# theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
# circle = np.c_[np.sin(theta), np.cos(theta)]
# noise_scale = np.random.uniform(size=circle.shape[0], low=0.90, high=1.10)
# noise_scale = np.c_[noise_scale, noise_scale]
# noise = np.random.uniform(size=(10, 2), low=-1, high=1)
# X = np.vstack((circle*noise_scale, noise))

# ## Plot the circle + noise 
# p = figure(width=400, height=200)
# p.scatter(X[:,0], X[:,1], color="blue")
# p.scatter(*noise.T, color="red")
# show(p)

# # %% [markdown]
# # ````{=html}
# # <!-- 

# # %%
# import line_profiler

# profile = line_profiler.LineProfiler()
# profile.add_function(mu_f.precompute)
# profile.enable_by_count()
# mu_f.precompute(R=R, w=w, normed=True, progress=True) 
# profile.print_stats(output_unit=1e-3)

# # %% [markdown]
# # alpha_thresholds = np.linspace(1e-12, max_scale*r, 100)
# # vine = np.vstack([ph(F(alpha)[1], engine="dionysus")[1] for alpha in alpha_thresholds])
# #
# # from bokeh.models import Range1d
# # p = figure_dgm(vine[-1,:])
# # p.scatter(np.ravel(vine['birth']), np.ravel(vine['death']))
# # p.x_range = Range1d(0, max_scale*r)
# # p.y_range = Range1d(0, max_scale*r)
# # show(p)
# # ``` -->
# # ````

