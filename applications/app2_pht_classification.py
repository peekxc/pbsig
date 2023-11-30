# %% [markdown] 
# title: Classification using the Persistent Homology transform
# author: Matt Piekenbrock
# date: 09/14/23

# %% Imports
#| echo: false
import numpy as np 
from pbsig import * 
from pbsig.linalg import * 
from pbsig.datasets import mpeg7
from pbsig.betti import *
from pbsig.persistence import *
from pbsig.simplicial import cycle_graph
from splex import *
import splex as sx
from scipy.spatial.distance import pdist, squareform
from splex.filters import generic_filter
import bokeh
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
output_notebook()


# %% Load dataset
# NOTE: PHT preprocessing centers and scales each S to *approximately* the box [-1,1] x [-1,1]
from pbsig.shape import stratify_sphere, normalize_shape
dataset = mpeg7(simplify=200)
X_, y_ = mpeg7(simplify=200, return_X_y=True)
print(dataset.keys())
dir_vect = stratify_sphere(1, 16)
as_2d = lambda x: x.reshape(len(x) // 2, 2)
X_ = np.array([np.ravel(normalize_shape(as_2d(x), dir_vect)) for x in X_])
shape_keys = list(dataset.keys())

def get_shape(shape_class: str, shape_id: int):
  return as_2d(X_[shape_keys.index((shape_class, shape_id)),:])

# %% Construct an interpolated operator 
from pbsig.pht import directional_transform
from pbsig.shape import normalize_shape, stratify_sphere
from pbsig.interpolate import ParameterizedFilter
from pbsig.betti import SpectralRankInvariant

S = cycle_graph(200-1)
ri = SpectralRankInvariant(S, p = 0)
ri.family = directional_transform(get_shape('turtle', 2), dv=32, normalize=False, nonnegative=1.0)
f_rng = np.array([(np.min(f(S)), np.max(f(S))) for f in ri.family])


from pbsig.apparent_pairs import apparent_pairs
f = next(iter(ri.family))
# apparent_pairs(S, f, p=0)


from pbsig.vis import figure_vineyard
dgms = ri.compute_dgms(S)
show(figure_vineyard(dgms, p = 0))

f_rng.min(), f_rng.max()

# %% Show various norms
from itertools import product 

shape_classes = ['turtle', 'bird', 'bone']
shape_ri = {}
for s_cls, s_id in product(shape_classes, range(1,8)):
  ri = SpectralRankInvariant(S, p = 0)
  ri.family = directional_transform(get_shape(s_cls, s_id), dv=32, normalize=False, nonnegative=1.0)
  f_rng = np.array([(np.min(f(S)), np.max(f(S))) for f in ri.family])
  f_min, f_max = f_rng[:,0].min(), f_rng[:,1].max()
  # birth, death = np.mean(f_rng[:,0])*1.1, np.mean(f_rng[:,1])*0.9
  f_pt = f_min + (f_max - f_min) / 2
  ri.sieve = [[-np.inf, f_pt, f_pt, np.inf]]
  ri.sift(w=0.0)
  shape_ri[(s_cls, s_id)] = ri


# %% Look at their heat trace signatures
from pbsig.dsp import phase_align

shape_colors = dict(zip(shape_classes, ['green', 'red', 'blue']))
p = figure(width=250, height=200)
for s_cls, s_id in product(shape_classes, [1,2,3,4,5,7]):
  s_sig = np.ravel(shape_ri[(s_cls, s_id)].summarize())
  p.line(np.arange(32), s_sig, color=shape_colors[s_cls])
show(p)

## Show phase-aligned signatures
mf = heat(t=50., complement=True)
p = figure(width=350, height=250)
for s_cls, s_id in product(shape_classes, [1,2,3,4,5,7]):
  s_ref = np.ravel(shape_ri[(s_cls, 1)].summarize(mf))
  s_sig = np.ravel(shape_ri[(s_cls, s_id)].summarize(mf))
  assert all(~np.isnan(s_sig))
  s_sig = phase_align(s_sig, s_ref)
  p.line(np.arange(32), sg(s_sig), color=shape_colors[s_cls])
show(p)

# %% Build a heat map




# %% 
from bokeh.models import Span
from pbsig.color import bin_color

theta = np.arange(132)/(2*np.pi)
theta_colors = bin_color(theta, "viridis")
ind = shape_keys.index(('turtle', 1))
trace_summary = f(shape_eigen[ind][:,55:]).sum(axis=1)
xs = list(pairwise(theta))
ys = list(pairwise(trace_summary))

def trace_figure(t: float):
  p = figure(width=600, height=250, title="Directional transform w/ heat trace")
  p.output_backend = 'svg'
  #p.line(theta, trace_summary, line_color=(theta_colors*255).astype(np.uint8))
  p.multi_line(xs, ys, line_color=(theta_colors*255).astype(np.uint8)[:-1], line_width=2.5)
  p.yaxis.visible = False
  p.toolbar_location = None
  p.xaxis.axis_label = r'Circle coordinate'
  it_line = Span(location=t, dimension='height', line_color='red', line_width=2)
  p.add_layout(it_line)
  return p 

from bokeh.io import export_svg
anim_path = "/Users/mpiekenbrock/pbsig/animations/trace_summary/"
for frame_id, t in enumerate(theta): 
  export_svg(
    trace_figure(t), 
    filename = anim_path + f"svgs/frame_{frame_id:03}.svg"
  )
show(p)

# %% 
from html2image import Html2Image
hti = Html2Image(output_path=anim_path+"frames/")
for frame_id in range(len(theta)):
  hti.screenshot(
    other_file = anim_path + 'svgs/' + f'frame_{frame_id:03}.svg', 
    save_as = f'frame_{frame_id:03}.png', 
    size = (600,250)
  )


# %%  Vineyards data 
from pbsig.vis import figure_dgm
DT = directional_transform(shape_xy, dv=132, normalize=False, nonnegative=1.0)
dgms = [ph(filtration(S, filter_f)) for filter_f in DT]
DT_theta = np.linspace(0, 2*np.pi, 132)

for i, filter_f in enumerate(DT):
  d = dgms[i][0]
  d['death'][np.isinf(d['death'])] = np.max(filter_f(faces(S,0)))
  dgms[i][0] = d

# %% Animation of a single transform
from bokeh.transform import linear_cmap
from bokeh.models import Arrow, NormalHead, Range1d, ColorBar, ColorMapper, ContinuousColorMapper, ColumnDataSource
from pbsig.color import bin_color
from more_itertools import collapse
from splex import lower_star_filter
from pbsig.vis import figure_dgm
from pbsig.color import rgb_to_hex
shape_xy = as_2d(X_[shape_keys.index(('turtle', 5))])
shape_xy[:,1] = -shape_xy[:,1] 

def figure_dt(S: ComplexLike, shape_xy: np.ndarray, t: float):
  """Creates the DT figure at some theta 't' in [0, 2*pi]"""
  theta = np.linspace(0, 2*np.pi, 512, endpoint=True)
  unit_circle = np.c_[np.cos(theta), np.sin(theta)]
  turbo_cm = ContinuousColorMapper(low=-1, high=1, palette="Turbo256")
  shape_diam = max(pdist(shape_xy))
  f_lb, f_ub = -shape_diam/2, shape_diam/2

  p = figure(width=400, height=425, title="Directional transform")
  p.output_backend = 'svg'
  p.x_range = Range1d(-2.0, 2.0)
  p.y_range = Range1d(-2.0, 2.0)
  p.line(*(unit_circle*1.5).T, color='black') 
  p.patch(*shape_xy.T, color='#d1d1d1a0')
  p.title.text_font_size = '18pt'

  dv = np.array([np.cos(t), np.sin(t)])
  filter_dv = lower_star_filter(shape_xy @ dv)
  line_xs = [shape_xy[e,0] for e in faces(S,1)]
  line_ys = [shape_xy[e,1] for e in faces(S,1)]
  dv_e_color = (bin_color(filter_dv(faces(S,1)), 'turbo', lb=f_lb, ub=f_ub) * 255).astype(np.uint8)
  p.multi_line(line_xs, line_ys, line_color = dv_e_color, line_width=3.5)

  source = ColumnDataSource(dict(x=shape_xy[:,0], y=shape_xy[:,1], fv=filter_dv(faces(S,0))))
  cmap = linear_cmap('fv', palette='Turbo256', low=f_lb, high=f_ub)
  sg = p.scatter(x='x', y='y', size=3.6, line_alpha=0.0, fill_color=cmap, source=source) #fill_color = dv_v_color
  nh = NormalHead(fill_color='black', size=10, fill_alpha=0.75, line_color='black')
  p.add_layout(Arrow(end=nh, line_color='black', line_width=2, x_start=dv[0]*1.2, y_start=dv[1]*1.2, x_end=dv[0]*1.8, y_end=dv[1]*1.8))
  color_bar = sg.construct_color_bar(padding=0)
  p.add_layout(color_bar, "below")
  p.toolbar_location = None

  circle_xs = list(pairwise(unit_circle[:,0]*1.5))
  circle_ys = list(pairwise(unit_circle[:,1]*1.5))
  source_circle = ColumnDataSource(dict(xs=circle_xs, ys=circle_ys, theta=np.linspace(0, 2*np.pi, len(circle_ys))))
  theta_cmap = linear_cmap('theta', palette='Viridis256', low=0, high=2*np.pi)
  ml = p.multi_line(xs='xs', ys='ys', line_width=3.5, line_color=theta_cmap, source=source_circle) 
  circle_color_bar = ml.construct_color_bar(padding=0) # p.add_layout(circle_color_bar, "below")
  p.axis.visible = False 

  ## Second plot: vineyard diagrams 
  theta_colors = bin_color(np.arange(132), 'viridis')
  j = np.searchsorted(DT_theta, t)

  q = figure_dgm(title="DT Vineyard", width=400, height=425)
  q.title.text_font_size = '18pt'
  q.output_backend = 'svg'
  q.x_range = Range1d(2.25, 5.25)
  q.y_range = Range1d(2.25, 5.25)
  for i, (dgm, theta) in enumerate(zip(dgms, DT_theta)):
    dv_color = rgb_to_hex(theta_colors[i,:]*255)
    dv_color = dv_color[:-2] + ('ff' if (i == j) else '20')
    q.scatter(dgm[0]['birth'], dgm[0]['death'], color=dv_color)
  q.toolbar_location = None
  q.axis.visible = False
  q.add_layout(circle_color_bar, "below")

  from bokeh.layouts import row, column
  return row(p, q)

# %% 
from bokeh.io import export_svg
anim_path = "/Users/mpiekenbrock/pbsig/applications/directional_transform_anim/"
for i, t in enumerate(DT_theta):
  dt_plot = figure_dt(S, shape_xy, t)
  export_svg(dt_plot, filename=anim_path + f"shape_dt_{i:03}.svg")

# %% 
from html2image import Html2Image
hti = Html2Image(output_path=anim_path+"frames/")
for frame_id in range(len(DT_theta)):
  hti.screenshot(
    other_file = anim_path + f'shape_dt_{frame_id:03}.svg', 
    save_as = f'frame_{frame_id:03}.png', 
    size = (800,425)
  )
# show(dt_plot)



# %% 




# %% Sample some boxes 
from pbsig.betti import Sieve
sieve = Sieve(S, p=0)
sieve.operators = LP
sieve.randomize_sieve(n_rect=5, plot = False)
sieve.sift(w=1.0)
sieve.summarize()

L = next(iter(sieve.operators))


# sieve.compute_dgms(S)
from pbsig.vis import figure_dgm
for filter_f in sieve.operators.family:
  K = filtration(S, filter_f)
  dgms = ph(K)
  show(figure_dgm(dgms[0]))



# %% Try the heat trace?
from pbsig.vis import figure_complex

p = figure_complex(S, pos = X, width = 250, height = 250, vertex_size=2, edge_line_width=5.5, edge_line_color="red")
show(p)
# np.array([HK.trace(LP.laplace(t), p=0.5, gram=False, method='eigenvalue') for t in np.linspace(0,1,32)])


# # %% Try the heat trace?
# from pbsig.linalg import HeatKernel
# from pbsig.utility import progressbar
# HK = HeatKernel(approx="mesh")

# hk_traces = []
# for data_key, X in progressbar(dataset.items()):
#   HK.fit(X=X, S=S, use_triangles=False)
#   hk_traces.append(HK.trace())

# ## Plot the heat traces for varying t 
# from pbsig.color import bin_color, colors_to_hex
# class_colors = bin_color(np.unique(y_), 'turbo')
# p = figure(width=250, height=200)
# for label, tr in zip(y_, hk_traces):
#   line_color = tuple((class_colors[label][:3]*255).astype(int))
#   p.line(np.arange(len(tr)), tr, line_color=line_color)
# show(p)

# # %% Show the heat trace on the parameterized laplacian for a single t
# heat_traces = []
# for theta in np.linspace(0, 1, 32):
#   L = LP.laplace(theta)
#   HK.fit(approx=L, use_triangles=False)
#   heat_traces.append(HK.trace())


# heat_traces = np.array(heat_traces)
# p = figure(width=250, height=200)
# p.line(np.arange(32), heat_traces[:,-8])
# p.scatter(np.arange(32), heat_traces[:,-8])
# show(p)














from numpy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh
from imate import trace, schatten, toeplitz
# A = toeplitz(20, 19, size=20, gram=True, format='csr')
# all((A == A.T).data)

# ew = eigvalsh(L.todense())
# n = L.shape[0]
# np.sum(np.sqrt(ew))/n ## nuclear norm of B where A = B^T B 

# p = 0.5
# # print((sum(ew) * (n**(-1/p))))
# print(sum(ew**p))
# print(trace(L, p=p, gram=False, method='eigenvalue')) # this one, but slq doesn't work
# trace(L, p=p, gram=True, method="eigenvalue") # close

schatten_norms = np.array([schatten(LP.laplace(t), p=1.0, method="exact") for t in np.linspace(0,1,32)])




A = L
sns.set(font_scale=1.15)
sns.set_style("white")
sns.set_style("ticks")
p = np.linspace(-0.01, 2.5, 201)
norm = np.zeros_like(p)
for i in range(p.size):
  norm[i] = schatten(A, p=p[i], method='eigenvalue')
norm0 = schatten(A, p=0, method='eigenvalue')
plt.semilogy(p, norm, color='black')
plt.semilogy(0, norm0, 'o', color='black')
plt.xlim([p[0]-0.5, p[-1]])
plt.ylim([0, 1e1])
plt.xlabel('$p$')
plt.ylabel('$\Vert \mathbf{A} \Vert_p$')
plt.title(r'Schatten Norm')
plt.show()


p = figure()
p.line(np.linspace(0,1,32), schatten_norms)
p.scatter(np.linspace(0,1,32), schatten_norms)
show(p)

# %% Show the graph filtering for different matrix functions across classes



# %%
# from pbsig.apparent_pairs import apparent_pairs
# from primate import slq
from pbsig.pht import directional_transform
from pbsig.betti import sample_rect_halfplane
dt = directional_transform(X, dv = 16)


rects = sample_rect_halfplane(5, lb = 0.0, ub = 1.001*max(pdist(X)), disjoint=True)



L = up_laplacian(S, p=0, form='array', weight=f)


# for f in dt:
#   L = up_laplacian(S, p=0, form='array', weight=f)
#   print(slq(L, matrix_function="heat", max_num_samples=50, error_atol=1e-10, error_rtol=1e-8, num_threads=6))



# %%
from pbsig.betti import Sieve
sieve = Sieve(S, dt)

from scipy.sparse.linalg import eigsh
eigsh(L, k=100, return_eigenvectors=False)


lower_star_filter((X @ np.array([0,1])) + radius)

# from scipy.sparse import eye
# from scipy.sparse.linalg import aslinearoperator
# cI = aslinearoperator(0.01*eye(L.shape[0]))

## Sample a couple rectangles in the upper half-plane and plot them 
R = sample_rect_halfplane(1, area=(0.10, 1.00))

## Plot all the diagrams, viridas coloring, as before
from pbsig.color import bin_color
E = np.array(list(S.faces(1)))
vir_colors = bin_color(range(64))
for ii, v in enumerate(uniform_S1(64)):
  dgm = ph0_lower_star(X @ v, E, max_death='max')
  plt.scatter(*dgm.T, color=vir_colors[ii], s=1.25)
  i,j,k,l = R[0,:]
  ij = np.logical_and(dgm[:,0] >= i, dgm[:,0] <= j) 
  kl = np.logical_and(dgm[:,1] >= k, dgm[:,1] <= l) 

