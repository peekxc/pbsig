# %% 
import numpy as np
import splex as sx
from ripser import ripser
from scipy.spatial.distance import pdist
from pbsig.persistence import as_dgm
from pbsig.vis import figure_dgm, show
from pbsig.betti import betti_query, BettiQuery
from pbsig.betti import BettiQuery
from math import comb
from pbsig.simplicial import complete_graph
from bokeh.io import output_notebook
output_notebook()

# %% 
theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
S1 = np.c_[np.cos(theta), np.sin(theta)]
Ellipsoid = lambda r1, r2: S1 @ np.diag([r1,r2])

S = sx.simplicial_complex(complete_graph(16), form="tree")
S.expand(2)
p = figure_dgm(ripser(Ellipsoid(1.0,1.0))['dgms'][1])
show(p)

# %% Vineyard across a range 
rng = np.linspace(0.1, 2, 30)
vine = []
for r in rng:
  vine.append(ripser(Ellipsoid(r,r))['dgms'][1])

from pbsig.color import bin_color
vine_pts = np.array(np.ravel(vine)).reshape((30, 2))
p = figure_dgm(None)
p.scatter(*vine_pts.T, color=bin_color(rng))
p.line(*vine_pts.T)
show(p)

# %% 
query = BettiQuery(S, p=1)
query.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else np.linalg.matrix_rank(L.todense())
query.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else np.linalg.matrix_rank(L.todense())

for r in rng:
  f = sx.flag_filter(pdist(Ellipsoid(r, r)))
  query.weights[0] = 1e-8 * np.ones(sx.card(S,0))
  query.weights[1] = f(sx.faces(S,1))
  query.weights[2] = f(sx.faces(S,2))
  print(f"Radius: ({r:.2f},{r:.2f}) Betti_1 = {query(i=0.5, j=1)}")

# %% Evaluate a grid
query.p_solver = lambda L: 0 if np.prod(L.shape) == 0 else np.linalg.matrix_rank(L.todense())
query.q_solver = lambda L: 0 if np.prod(L.shape) == 0 else np.linalg.matrix_rank(L.todense())
R1, R2 = np.meshgrid(rng, rng)
results = []
for ii, (r1, r2) in enumerate(zip(np.ravel(R1), np.ravel(R2))):
  f = sx.flag_filter(pdist(Ellipsoid(r1, r2)))
  query.weights[0] = 1e-8 * np.ones(sx.card(S,0))
  query.weights[1] = f(sx.faces(S,1))
  query.weights[2] = f(sx.faces(S,2))
  results.append(query(i=0.5, j=1))
  if ii % 20 == 0:
    print(ii)

# %% Show the rank-valued objective surface
import plotly as plt
import plotly.graph_objects as go
z_rank = np.array(results).reshape(R1.shape)
fig = go.Figure(data=[go.Surface(x=rng, y=rng, z=z_rank)])
fig.update_layout(
  title='Multiplicity', autosize=True,
  width=500, height=500
)
fig.show()

# %% Figure out the parameterization that leads to a good surface in 2d 



# %% Try the tikhonov w/ positive sign width 
from primate.trace import hutch
query.sign_width = 0.0
def patch(A):
  if all(A.row == A.col) and all(A.col == np.arange(A.shape[0])):
    return np.sum(A.data / (A.data + 1e-2))
  return hutch(A, fun=lambda x: x / (x + 1e-2), atol=0.10)
query.p_solver = patch #lambda L: hutch(L, fun=lambda x: x / (x + 1e-11), atol=0.10)
query.q_solver = patch #lambda L: hutch(L, fun=lambda x: x / (x + 1e-11), atol=0.10)

results = []
for ii, (r1, r2) in enumerate(zip(np.ravel(R1), np.ravel(R2))):
  f = sx.flag_filter(pdist(Ellipsoid(r1, r2)))
  query.weights[1] = f(sx.faces(S,1))
  query.weights[2] = f(sx.faces(S,2))
  # query(i=1.2, j=1.2)
  results.append(query(i=0.5, j=1))
  if ii % 20 == 0:
    print(ii)


# A = query.operator(3, i=0.5, j=1, deflate=True)
# hutch(A, fun=lambda x: x / (x + 1e-11), atol=0.10)
# query.q_solver(A)

# %% Show the raw objective surface in 3-D
z = np.array(results).reshape(R1.shape)
fig = go.Figure(data=[go.Surface(x=rng, y=rng, z=z)])
fig.update_layout(
  title='Multiplicity (Tikhonov)', autosize=True,
  width=500, height=500
)
fig.show()

# %% Smooth with golay 
import sgolay2
z = np.array(results).reshape(R1.shape)
zs = sgolay2.SGolayFilter2(window_size=7, poly_order=1)(z)
fig = go.Figure(data=[go.Surface(x=rng, y=rng, z=zs)])
fig.update_layout(
  title='Multiplicity (Tikhonov)', autosize=True,
  width=500, height=500
)
fig.show()

# %% Contour in 2D
from bokeh.plotting import figure, show
zs = sgolay2.SGolayFilter2(window_size=7, poly_order=1)(z)
levels = np.linspace(np.min(zs), np.max(zs), 32)
levels = np.quantile(levels, [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0])
level_colors = (bin_color(levels, "inferno") * 255).astype(np.uint8)

from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE)

from bokeh.models import PrintfTickFormatter
formatter = PrintfTickFormatter(format='%.1f')
p = figure(width=400, height=300, x_range=(np.min(rng), np.max(rng)), y_range=(np.min(rng), np.max(rng)), title="Persistent rank (Tikhonov)")
contour_r = p.contour(rng, rng, zs, levels=levels, fill_color=level_colors[:-1], line_color="black")
cbar = contour_r.construct_color_bar(margin=15, formatter = formatter)
p.add_layout(cbar, "left")
# p.xaxis.axis_label = "R1"
# p.yaxis.axis_label = "R2"
p.toolbar_location = None
# p.title = r"\[Ellipsoid\]"
show(p)
# fig = go.Figure(
#   data=go.Contour(z=zs, x=rng, y=rng, contours=dict(start=np.min(zs), end=np.max(zs), size=16)), 
#   layout=go.Layout(autosize=False,width=500, height=500)
# )
# fig.show()

# %% Put them all together 
from pbsig.vis import figure_complex

f = sx.flag_filter(pdist(Ellipsoid(1.1, 1.1)))
S_a = sx.simplicial_complex([s for s in S if f(s) <= 0.5], form="tree")
S_b = sx.simplicial_complex([s for s in S if f(s) <= 1.0], form="tree")
qa1 = figure_complex(S_a, pos=Ellipsoid(1.1, 1.1), width=150, height=150, toolbar_location=None)
qb1 = figure_complex(S_b, pos=Ellipsoid(1.1, 1.1), width=150, height=150, toolbar_location=None)

f = sx.flag_filter(pdist(Ellipsoid(1.4, 1.4)))
S_a = sx.simplicial_complex([s for s in S if f(s) <= 0.5], form="tree")
S_b = sx.simplicial_complex([s for s in S if f(s) <= 1.0], form="tree")
qa2 = figure_complex(S_a, pos=Ellipsoid(1.4, 1.4), width=150, height=150, toolbar_location=None, title="K_a at ")
qb2 = figure_complex(S_b, pos=Ellipsoid(1.4, 1.4), width=150, height=150, toolbar_location=None)

from bokeh.layouts import row, column
p.scatter(x=1.1, y=1.1, color="green")
show(row(p, column(row(qa1, qb1), row(qa2, qb2))))



