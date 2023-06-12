# ---
# title: Circle optimization of varying density
# format: html
# editor:
#   render-on-save: true
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# X = noisy_circle(70, n_noise=30, perturb=0.15)
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

p = figure(width=200, height=200, match_aspect=True,  title="Scatter by density")
p.scatter(*X.T, color=bin_color(x_density), size=8)
q = figure(width=200, height=200, match_aspect=True,  title="Scatter by codensity")
q.scatter(*X.T, color=bin_color(x_codensity), size=8)
r = figure_complex(S, pos=X, color=x_codensity, title="Complex by codensity", simplex_kwargs={0: {'size': 8}}, width=200, height=200)
s = figure_complex(S, pos=X, color=s_diam, title="Complex by diameter", simplex_kwargs={0: {'size': 8}}, width=200, height=200) # bin_kwargs={'scaling': 'equalize'})
show(row(p, q, r, s))


# %% [markdown]
# ## Generate a parameterized family of weight functions
#
# Given a fixed simplicial complex $K$ with size $N = \lvert K \rvert$, the following function generates a parameterized family $\{f_\alpha(K)\}_{\alpha \in \mathbb{R}}$
#
# $$ 
# \begin{align*}
# f \; : \; &\alpha &\to & \; \mathbb{R}_{+}^{N}  \\
#     & \mathbb{R} &\mapsto & \; f_\alpha(K)
# \end{align*}
# $$
#
# where $\alpha \in \mathbb{R}$ parameterizes the bandwidth of a kernel-density estimate on a point $X \subset \mathbb{R}^n$ from which $K$ was constructed.  

# %% Parameterized the weight function
import functools
def codensity(alpha: float):
  """Codensity of a point set _X_ with some bandwidth _alpha_."""
  x_density = gaussian_kde(X.T, bw_method=alpha).evaluate(X.T)
  x_density /= max(x_density)
  x_codensity = max(x_density) - x_density
  return functools.partial(lambda s, c: max(c[s]), c=x_codensity)


# %% Change density parameter
figures = []
for a in np.linspace(0.05, 0.70, 6): 
  p = figure(width=200, height=200, title=f"Density: {a:.2f}")
  f = codensity(a)
  x_codensity = np.array([f(Simplex(i)) for i in range(X.shape[0])])
  p.scatter(*X.T, color=bin_color(x_codensity, lb=0, ub=0.70), size=8)
  figures.append(p)
show(row(figures))

# %% Show three plots
from bokeh.models import Range1d
from pbsig.vis import figure_complex, figure_dgm
complex_figures, dgm_figures = [], []
for a in [0.10, 0.414, 0.75]: 
  p = figure(width=250, height=250, title=f"Density: {a:.2f}")
  f = codensity(a)
  s_codensity = np.array([f(s) for s in faces(S)])
  p = figure_complex(S, pos=X, color=s_codensity, title="Complex by codensity", simplex_kwargs={0: {'size': 8}}, width=250, height=250)
  # ?p.scatter(*X.T, color=bin_color(x_codensity, lb=0, ub=0.70), size=9)
  p.toolbar_location = None
  complex_figures.append(p)
  K = filtration(S, f)
  q = figure_dgm(ph(K, engine="dionysus")[1], pt_size=10, width=250, height=250)
  q.x_range = Range1d(0,0.90)
  q.y_range = Range1d(0,0.90)
  q.toolbar_location = None
  R = (0.2, 0.4, 0.6, 0.8)
  r_width, r_height = R[1]-R[0], R[3]-R[2]
  q.rect(R[0]+r_width/2, R[2]+r_height/2, r_width, r_height, color="orange", alpha=1.0, fill_alpha=0.0, line_width=2.0)
  dgm_figures.append(q)


show(column(row(complex_figures), row(dgm_figures)))
# export_png(column(row(complex_figures), row(dgm_figures)), filename="/Users/mpiekenbrock/pbsig/notes/presentation/codensity_pers_ex.png")

# %% single plot 
f = codensity(0.404)
s_codensity = np.array([f(s) for s in S])
p = figure_complex(S, pos=X, color=s_codensity, title="Complex with optimal codensity", simplex_kwargs={0: {'size': 8}}, width=400, height=400)
p.toolbar_location = None
show(p)


# %% [markdown]
# ## Vineyards

# %% Compute the the vineyards
alpha_family = np.linspace(0.15, 0.70, 100)
codensity_family = [codensity(a) for a in alpha_family]
dgms = []
for f in codensity_family:
  K = filtration(S, f)
  dgms.append(ph(K, engine="dionysus")[1])
family_dgms = np.vstack([np.array((np.repeat(i, d.shape[0]), d['birth'], d['death'])).T for i,d in enumerate(dgms)])

# %% Plot the vineyards
p = figure_dgm(x_range = (0, 1), y_range = (0, 1))
p.scatter(family_dgms[:,1], family_dgms[:,2], color=bin_color(family_dgms[:,0]))
show(p)


# %% Verify multiplicity matches with persistence diagram visually
from pbsig.betti import MuFamily
R = (0.2, 0.4, 0.6, 0.8)
mu_f = MuFamily(S, codensity_family, p = 1, form='array')
mu_f.precompute(R, w=1.30, normed=True)

# %% Compare
p = figure_dgm(x_range=(0, 1), y_range=(0, 1), width=300, height=300)
p.scatter(family_dgms[:,1], family_dgms[:,2], color=bin_color(family_dgms[:,0]))
r_width, r_height = R[1]-R[0], R[3]-R[2]
p.rect(R[0]+r_width/2, R[2]+r_height/2, r_width, r_height, alpha=1.0, fill_alpha=0.0)
q = figure(width=300, height=300, title="Multiplicity")
q.step(alpha_family, mu_f(smoothing=None))
q.step(alpha_family, mu_f(smoothing=True), color='red')
p.toolbar_location = None 
q.toolbar_location = None
q.xaxis.axis_label = r"$$\alpha$$"
show(row(p,q))


top10  = lambda x: np.append(np.sort(x)[-10:], [0]*(len(x)-10))

# %% Look at the other relaxations
from scipy.signal import savgol_filter
from pbsig.linalg import *
mu_rk = mu_f(smoothing=None, terms=False)
mu_nn = mu_f(smoothing=False, terms=False)
mu_sa = mu_f(smoothing=sgn_approx(eps=5.0, p=2.8), terms=False) # 0.008, 2.8
mu_sa = savgol_filter(mu_sa, 8, 3)

mu_sa2 = mu_f(smoothing=sgn_approx(eps=0.3, p=2.8), terms=False) # 0.008, 2.8
mu_sa2 = savgol_filter(mu_sa2, 8, 3)

mu_sa3 = mu_f(smoothing=sgn_approx(eps=0.01, p=2.8), terms=False) # 0.008, 2.8
mu_sa3 = savgol_filter(mu_sa3, 8, 3)

mu_sa4 = mu_f(smoothing=sgn_approx(eps=0.005, p=2.8), terms=False) # 0.008, 2.8
mu_sa4 = savgol_filter(mu_sa4, 8, 3)

mu_sa5 = mu_f(smoothing=sgn_approx(eps=0.001, p=2.8), terms=False) # 0.008, 2.8
mu_sa5 = savgol_filter(mu_sa5, 8, 3)

mu_sa6 = mu_f(smoothing=sgn_approx(eps=0.0001, p=2.8), terms=False) # 0.008, 2.8
# mu_sa6 = savgol_filter(mu_sa5, 8, 3)

p = figure(width=550, height=300, title="Multiplicity")
p.step(alpha_family, mu_rk, line_color="black", legend_label="rank")
# p.line(alpha_family, mu_nn, line_color="red", legend_label="nuclear")
p.line(alpha_family, mu_sa, line_color="red", legend_label="nuclear")
p.line(alpha_family, mu_sa2, line_color="orange", legend_label="ht (0.30)")
p.line(alpha_family, mu_sa3, line_color="green", legend_label="ht (0.10)")
p.line(alpha_family, mu_sa4, line_color="blue", legend_label="ht (0.05)")
p.line(alpha_family, mu_sa5, line_color="gray", legend_label="ht (0.01)")
# p.line(alpha_family, mu_sa6, line_color="darkgray", legend_label="ht (0.001)")


p.toolbar_location = None
# p.x_range = Range1d()
# show(p)

# %% 
dom = np.linspace(-1,1,1500)
q = figure(width=375, height=300, title="Sign approx. (φ)")
# q.line(dom, [1 if x != 0 else 0 for x in dom ], color='black')
q.line([-1,0.001], [1,1], color='black', legend_label="sgn+")
q.line([0.001,1], [1,1], color='black')
q.scatter([0], [0], color='black', size=5)
q.line(dom, np.abs(dom)**1.0, color="red", legend_label="l1")
q.line(dom, sgn_approx(eps=0.3, p=1.0)(np.abs(dom)), color="orange", legend_label="φ (0.30)")
q.line(dom, sgn_approx(eps=0.10, p=1.0)(np.abs(dom)), color="green", legend_label="φ (0.10)")
q.line(dom, sgn_approx(eps=0.05, p=1.0)(np.abs(dom)), color="blue", legend_label="φ (0.05)")
q.line(dom, sgn_approx(eps=0.01, p=1.0)(np.abs(dom)), color="gray", legend_label="φ (0.01)")
# q.line(dom, sgn_approx(eps=0.001, p=1.0)(np.abs(dom)), color="darkgray", legend_label="φ (0.001)")
q.x_range = Range1d(-1.8,1.0)
q.y_range = Range1d(-0.05,1.05)
# show(q)
q.legend.location = 'bottom_left'
q.toolbar_location = None 
show(row(p,q))


# %% 
min_supp = min(alpha_family[mu_sa > 0.0001])
max_supp = max(alpha_family[mu_sa > 0.0001])


alpha_family[np.argmax(mu_sa)]

## Just rn optimization
from scipy.optimize import minimize


# %% Constitutive terms
mu_rk = mu_f(smoothing=None, terms=True).T
mu_nn = mu_f(smoothing=False, terms=True).T

#mu_nn = mu_f(smoothing=lambda x: sum(abs(x))/max(abs(x)), terms=True).T
mu_sa = mu_f(smoothing=sgn_approx(eps=1e-1, p=1.0), terms=True).T

f1,f2,f3 = figure(width=400, height=int(500/3), title="Rank"), figure(width=400, height=int(500/3), title="Nuclear norm"), figure(width=400, height=int(500/3), title="Sgn approximation (0.1)")
f1.toolbar_location = None 
f2.toolbar_location = None 
f3.toolbar_location = None 
figures = []
for i in range(4):
  p = figure(width=300, height=250, title=f"Multiplicity terms (T{i+1})")
  p.step(alpha_family, mu_rk[:,i], line_color="black")
  p.line(alpha_family, mu_nn[:,i], line_color="red")
  p.line(alpha_family, mu_sa[:,i], line_color="blue")
  p.toolbar_location = None
  figures.append(p)

mu_nn = mu_f(smoothing=sgn_approx(eps=5.0, p=2.8), terms=True).T
f1.step(alpha_family, mu_rk.sum(axis=1), color="black")
f1.line(alpha_family, mu_rk[:,0]/np.max(abs(mu_rk[:,0])), line_dash='dashed', color="black")
f1.line(alpha_family, mu_rk[:,1]/np.max(abs(mu_rk[:,1])), line_dash='dotted', color="black")
f1.line(alpha_family, mu_rk[:,2]/np.max(abs(mu_rk[:,2])), line_dash='dotted', color="black")
f1.line(alpha_family, mu_rk[:,3]/np.max(abs(mu_rk[:,3])), line_dash='dashed', color="black")

f2.step(alpha_family, mu_nn.sum(axis=1), color="red")
f2.line(alpha_family, mu_nn[:,0]/np.max(abs(mu_nn[:,0])), line_dash='dashed', color="red")
f2.line(alpha_family, mu_nn[:,1]/np.max(abs(mu_nn[:,1])), line_dash='dotted', color="red")
f2.line(alpha_family, mu_nn[:,2]/np.max(abs(mu_nn[:,2])), line_dash='dotted', color="red")
f2.line(alpha_family, mu_nn[:,3]/np.max(abs(mu_nn[:,3])), line_dash='dashed', color="red")

f3.step(alpha_family, mu_sa.sum(axis=1), color="blue")
f3.line(alpha_family, mu_sa[:,0]/np.max(abs(mu_sa[:,0])), line_dash='dashed', color="blue")
f3.line(alpha_family, mu_sa[:,1]/np.max(abs(mu_sa[:,1])), line_dash='dotted', color="blue")
f3.line(alpha_family, mu_sa[:,2]/np.max(abs(mu_sa[:,2])), line_dash='dotted', color="blue")
f3.line(alpha_family, mu_sa[:,3]/np.max(abs(mu_sa[:,3])), line_dash='dashed', color="blue")

show(
  row(column(row(figures[:2]), row(figures[2:])), column(f1,f2,f3))
)

# %% Gradient calculation
from pbsig.betti import MuQuery
import numdifftools as nd

mu_q = MuQuery(S, p=1, R=R, w=0.30)
mu_q.choose_solver(solver="dac", normed=True, form="array")
# mu_f.choose_solver(solver="irl", normed=True, tol=1e-7, form="array")
# mu_f.choose_solver(solver="irl", normed=True, tol=1e-7, form="lo")
mu_q.smoothing = sgn_approx(eps=1e-2, p=2.0) ## STOP AND SHOW 1e-1 and 1e-2
print(mu_q.solver.params)

## Continuous parameterization 
codensity_mu = lambda alpha: mu_q(f=codensity(alpha))
codensity_d = nd.Derivative(codensity_mu, n=1, order=2, method="central", richardson_terms=2)


# %% Takes 10 seconds to evaluate a single derivative!
from scipy.sparse.linalg import eigsh
from splex.sparse import boundary_matrix, _fast_boundary
from pbsig.betti import update_weights
from pbsig.linalg import rank_bound, trace_threshold
import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(rank_bound)
profile.add_function(mu_q.__call__)
# if mu_q.solver.solver.__module__ ==  'numpy.linalg':
#   profile.add_function(mu_q.solver.solver.__wrapped__)  # for numpy 
profile.enable_by_count()
codensity_d(0.05)
profile.print_stats(output_unit=1e-3, stripzeros=True)

# %% Inspect the smoothing choice
mu_f = MuFamily(S, codensity_family, p = 1)
mu_f.precompute(R, w=0.30, normed=True)

# %% Evaluate correctness of gradient
codensity_mu = lambda alpha: mu_q(f=codensity(alpha))
codensity_d = nd.Derivative(codensity_mu, n=1, order=2, method="central", richardson_terms=2)
mu_grads = np.array([codensity_d(alpha) for alpha in alpha_family])

# %% Instability of the gradient calculation
mu_rk = mu_f(smoothing=None, terms=False)
mu_sa = mu_f(smoothing=mu_q.smoothing, terms=False)
p = figure(width=300, height=250, title="Multiplicity")
p.step(alpha_family, mu_rk, line_color="black")
p.line(alpha_family, mu_sa, line_color="blue")
p.scatter(alpha_family, mu_sa, color=np.where(mu_grads >= 0, "red", "blue"))
show(p)

# %%
## Lipshitz constant 
mu_sa_terms = mu_f(smoothing=mu_q.smoothing, terms=True)
f_diffs = abs(np.diff(mu_sa_terms[2,:])) / abs(np.diff(alpha_family))
Lc = max(f_diffs)
print(f"Estimated Lipshitz constant: {Lc}")
print(np.histogram(f_diffs))


# %%
mu_q.terms = True
codensity_mu = lambda alpha: mu_q(f=codensity(alpha))
codensity_d = nd.Derivative(codensity_mu, n=1, order=2, method="central", richardson_terms=2)
mu_grad_terms = np.array([codensity_d(alpha) for alpha in alpha_family])

# %% Inspect whether the individual terms are unstable
j = 0
mu_rk_terms = mu_f(smoothing=None, terms=True)
mu_sa_terms = mu_f(smoothing=mu_q.smoothing, terms=True)
p = figure(width=400, height=400, title="Multiplicity Term 1")
p.step(alpha_family, mu_rk_terms[j,:], line_color="black")
p.line(alpha_family, mu_sa_terms[j,:], line_color="blue")
p.scatter(alpha_family, mu_sa_terms[j,:], color=np.where(mu_grad_terms[:,j] >= 0, "red", "blue"))
show(p)

# %% Derivative testing
# from pbsig.betti import MuQuery
# mu_q = MuQuery(S, p=1, R=R, w=0.30)
# mu_q.choose_solver(solver="dac", normed=True, form="array")
# mu_q.smoothing = sgn_approx(eps=1e-2, p=2.0) 

# ## (1) alpha |-> weight on every simplex
# codensity_d = mu_q._grad_scalar_product(codensity)

# ## (2) weight function |-> Up Laplacian 



# %% proxop
# import proxop 
# om = proxop.Root(0.2)

# p = figure(width=200, height=100)
# dom = np.linspace(-1,1,1000)
# p.line(dom, [om(xi) for xi in dom])
# q = figure(width=200, height=100)
# q.line(dom, np.ravel([np.sign(xi)*np.sign(om.prox(xi, gamma=0.1))*om.prox(xi, gamma=0.1) for xi in dom]))
# show(row(q,p))


# %%
