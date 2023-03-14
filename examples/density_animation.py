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
from more_itertools import chunked
from pbsig.utility import progressbar
from pbsig.vis import bin_color
from pbsig.vis import figure_complex

output_notebook()

# %% Load data 
from pbsig.datasets import noisy_circle
from splex.geometry import delaunay_complex
np.random.seed(1234)
X = noisy_circle(80, n_noise=30, perturb=0.15)
S = delaunay_complex(X)

# %% Create filtering function
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist

def codensity(bw: float):
  x_density = gaussian_kde(X.T, bw_method=bw).evaluate(X.T)
  x_codensity = max(x_density) - x_density
  return x_codensity

## TODO: make vectorized ! and don't assume vertices are labeled 0...(n-1)
def lower_star_weight(x: ArrayLike) -> Callable:
  def _weight(s: SimplexConvertible) -> float:
    return max(x[s])
  return _weight

from splex import *
from pbsig.persistence import ph
from pbsig.vineyards import update_lower_star
x_codensity = codensity(0.15)
K = filtration(S, f = lambda s: max(x_codensity[s]))


# %% 
R,V = ph(K, p=None, output="RV", engine="python", validate=False)
B = boundary_matrix(K)

## Make a spy plot 
def figure_spy(A: spmatrix, highlight = None, **kwargs):
  from bokeh.models import Span
  highlight = highlight if highlight is not None else []
  n = A.shape[0]
  p = figure(width=400, height=400, toolbar_location=None, **kwargs)
  rect_coords = np.array([(j,i,4,4) for i,j in zip(*A.nonzero())])
  p.rect(*rect_coords.T, color="black")
  for h in highlight: 
    s = Span(location=h, dimension='height',line_color='red', line_width=0.80)
    p.add_layout(s)
    s = Span(location=h, dimension='width',line_color='red', line_width=0.80)
    p.add_layout(s)
    s = Span(location=h+1, dimension='height',line_color='orange', line_width=0.80)
    p.add_layout(s)
    s = Span(location=h+1, dimension='width',line_color='orange', line_width=0.80)
    p.add_layout(s)
  # rect_coords = np.array([(j,i,8,8) for i,j in zip(*A.nonzero()) if j in highlight])
  # p.rect(*rect_coords.T, color="red")
  p.x_range.range_padding = p.y_range.range_padding = 0
  p.y_range.flipped = True
  p.x_range.flipped = False
  p.axis.visible = False
  p.xgrid.grid_line_color = None
  p.ygrid.grid_line_color = None
  return p 


# S = []
# for alpha in np.linspace(0.15, 0.70, 10):
#   f = lower_star_weight(codensity(alpha))
#   L = filtration(K.values(), f = f)
#   schedule, _ = linear_homotopy(K, L)
#   K.reindex(f)
#   S.append(schedule)
#   print("alpha: ", alpha)

#from more_itertools import flatten
#S = np.fromiter(flatten(S), dtype=int)
#np.savetxt('/Users/mpiekenbrock/pbsig/animations/density_vineyards/schedule.txt', S, fmt='%d', delimiter=' ')
schedule = np.loadtxt('/Users/mpiekenbrock/pbsig/animations/density_vineyards/schedule.txt', dtype=np.uint32, delimiter=" ")
nc = int(len(schedule)/2400)
alpha_family = np.linspace(0.15, 0.70, int(np.ceil(len(schedule)/nc)))

# nc = int(comb(len(K),2)/(1200*2)) ## 20 seconds @ 120 fps == 1200 frames wanted 

schedule_it = progressbar(transpose_rv(R, V, schedule), count=len(schedule))
for i,(s,h) in enumerate(zip(chunked(schedule_it, nc), chunked(schedule, nc))): 
  p = figure_spy(V, highlight=h)
  export_png(p, filename=f"/Users/mpiekenbrock/pbsig/animations/density_vineyards/{i}.png", width=400, height=400, timeout=3)
  
nc = int(len(schedule)/2400)
x_codensity = codensity(0.15)
K = filtration(S, f = lambda s: max(x_codensity[s]), form="set")
simplices = list(faces(K))

for i, (alpha, ind) in progressbar(enumerate(zip(alpha_family, chunked(schedule, nc))), count=len(alpha_family)):
  ind = np.array(ind)
  changed_simplices = set([])
  for ii in ind:
    c,t = simplices[ii], simplices[ii+1]
    changed_simplices |= set([c,t])
    simplices[ii] = t
    simplices[ii+1] = c
  f = lower_star_weight(codensity(alpha))
  s_color = bin_color(np.array([f(s) for s in faces(K)]))
  s_color_red = np.array([s_color[i,:] if s not in changed_simplices else [1.0, 0.0, 0.0, 1.0] for i, s in enumerate(faces(K))])
  p = figure_complex(K, pos=X, color=s_color_red, title="Complex by codensity", simplex_kwargs={0: {'size': 8}}, width=400, height=400)
  p.toolbar_location = None
  export_png(p, filename=f"/Users/mpiekenbrock/pbsig/animations/density_vineyards/complex/{i}.png", width=400, height=400, timeout=15)
  # show(p)


## Export the vineyard 
dgms = []
alpha_family = np.linspace(0.15, 0.70, int(np.ceil(len(schedule)/nc)))
for i, alpha in progressbar(enumerate(alpha_family), count=len(alpha_family)):
  f = lower_star_weight(codensity(alpha))
  K.reindex(f)
  dgms.append(ph(K, engine="dionysus")[1])
vineyard = np.array([(float(d['birth']), float(d['death'])) for d in dgms])
# np.savetxt(fname="/Users/mpiekenbrock/pbsig/animations/density_vineyards/dgms/vine.txt", X=vineyard)
vineyard = np.loadtxt("/Users/mpiekenbrock/pbsig/animations/density_vineyards/dgms/vine.txt")

from pbsig.vis import figure_dgm, bin_color
from bokeh.models import Range1d
for i, pt in progressbar(enumerate(vineyard), count=len(vineyard)):
  f = lower_star_weight(codensity(alpha))
  s_color = bin_color(np.array([f(s) for s in faces(K)]))
  s_color_red = np.array([s_color[i,:] if s not in changed_simplices else [1.0, 0.0, 0.0, 1.0] for i, s in enumerate(faces(K))])
  p = figure_complex(K, pos=X, color=s_color_red, title="Complex by codensity", simplex_kwargs={0: {'size': 8}}, width=400, height=400)
  p.toolbar_location = None
  p.x_range = Range1d(0, 0.70)
  p.y_range = Range1d(0, 0.70)
  p.scatter(pt[0], pt[1], color="red")
  p.toolbar_location = None
  p.line([pt[0],pt[0]], [pt[0],pt[1]], line_color='red', line_width=1.5)
  p.line([0,pt[0]], [pt[1],pt[1]], line_color='red', line_width=1.5)
  #show(p)
  export_png(p, filename=f"/Users/mpiekenbrock/pbsig/animations/density_vineyards/dgms/{i}.png", width=400, height=400, timeout=15)
  # show(p)


from pbsig.vis import figure_complex, bin_color
from bokeh.models import Range1d
for i, (alpha, ind) in progressbar(enumerate(zip(alpha_family, chunked(schedule, nc))), count=len(alpha_family)):
  f = lower_star_weight(codensity(alpha))
  s_color = bin_color(np.array([f(s) for s in faces(K)]))
  p = figure_complex(K, pos=X, color=s_color, title="Complex by codensity", simplex_kwargs={0: {'size': 8}}, width=400, height=400)
  p.toolbar_location = None
  #show(p)
  export_png(p, filename=f"/Users/mpiekenbrock/pbsig/animations/density_vineyards/complex_plain/{i}.png", width=400, height=400, timeout=15)


  # s_color = bin_color(np.array([f(s) for s in faces(K)]))
  # s_color_red = np.array([s_color[i,:] if s not in changed_simplices else [1.0, 0.0, 0.0, 1.0] for i, s in enumerate(faces(K))])
# changed_vertices = [c for c in changed_simplices if len(c) == 1]
# changed_edges = [c for c in changed_simplices if len(c) == 2]
# changed_triangles = [c for c in changed_simplices if len(c) == 3]
# changed_vertices = np.array(list(set(changed_vertices)), dtype=np.uint16)
# changed_edges = np.array(list(set(changed_edges)), dtype=np.uint16)
# changed_triangles = np.array(list(set(changed_triangles)), dtype=np.uint16)


# p.y_range = Range1d(0,-10)
# p.x_range = Range1d(0,10)

# p.background_fill_color = "#000000"
# p.image_rgba(image=[np.flipud(D)], x=0, y=0, dw=10, dh=10, dilate=True, coor='blue')

# update_lower_star(K, R, V, lower_star_weight(codensity(0.10)), vines=True, progress=True)
# export_png(plot, filename="plot.png")