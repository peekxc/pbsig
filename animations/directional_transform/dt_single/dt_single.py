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
from scipy.spatial.distance import pdist, squareform
from splex.Simplex import filter_weight
import bokeh
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.models import Arrow, NormalHead, Range1d, ColorBar, ColorMapper, ContinuousColorMapper, ColumnDataSource
from pbsig.color import bin_color
from more_itertools import collapse
from splex import lower_star_weight
from pbsig.vis import figure_dgm
from pbsig.color import rgb_to_hex
from pbsig.linalg import ParameterizedLaplacian
from pbsig.pht import directional_transform
from pbsig.shape import normalize_shape, stratify_sphere
output_notebook()

# from selenium.webdriver.chrome.webdriver import WebDriver as Chrome
# Chrome()

# %% Load dataset
# NOTE: PHT preprocessing centers and scales each S to *approximately* the box [-1,1] x [-1,1]
dataset = mpeg7(simplify=200)
X_, y_ = mpeg7(simplify=200, return_X_y=True)
print(dataset.keys())
dir_vect = stratify_sphere(1, 16)
as_2d = lambda x: x.reshape(len(x) // 2, 2)
X_ = np.array([np.ravel(normalize_shape(as_2d(x), dir_vect)) for x in X_])
shape_keys = list(dataset.keys())

# %% Construct an interpolated operator 
S = cycle_graph(200-1)
LP = ParameterizedLaplacian(S, p=0, form="lo", normed=True) ## Make a parameterized laplacian over a *fixed* complex

## Choose a data set to parameterize a directional transform over
# LP.family = directional_transform(as_2d(X_[shape_keys.index(('bird', 1)),:]), dv=32, normalize=False, nonnegative=1.0)
# LP.interpolate_family(interval=[0, 1])

# %% directional transform single
def figure_dt(S: ComplexLike, shape_xy: np.ndarray, t: float):
  """Creates the DT figure at some theta 't' in [0, 2*pi]"""
  theta = np.linspace(0, 2*np.pi, 512, endpoint=True)
  unit_circle = np.c_[np.cos(theta), np.sin(theta)]
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
  filter_dv = lower_star_weight(shape_xy @ dv)
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
  p.axis.visible = False 
  return p

# show(figure_dt(S, shape_xy, 0.0))

# %% Animation of a single transform
from bokeh.io import export_svg
DT_theta = np.linspace(0, 2*np.pi, 132)
shape_xy = X_[shape_keys.index(('turtle', 5))]
shape_xy = shape_xy.reshape(len(shape_xy) // 2, 2)
shape_xy[:,1] = -shape_xy[:,1] 
anim_path = "/Users/mpiekenbrock/pbsig/animations/directional_transform_anim/dt_single/"
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
    size = (400,425)
  )
# show(dt_plot)

