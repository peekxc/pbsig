
import numpy as np
from itertools import * 
import networkx as nx
from typing import * 

from .combinatorial import inverse_choose
from .color import *
from .linalg import *

import bokeh 
from bokeh.plotting import figure, show
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import GraphRenderer, Ellipse, Range1d, Circle, ColumnDataSource, MultiLine, Label, LabelSet, Button, Span
from bokeh.palettes import Spectral8, RdGy
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.io import output_notebook, show, save
from bokeh.transform import linear_cmap
from bokeh.layouts import column


def plot_dgm(dgm, pt_size: int = 5, show_filter: bool = False):
  output_notebook(verbose=False, hide_banner=True)
  max_val = max(dgm["death"], key=lambda v: v if v != np.inf else -v) 
  max_val = (max_val if max_val != np.inf else 1.0)
  min_val = min(dgm["birth"])
  min_val = (min_val if min_val != 1.0 else 0.0)
  delta = abs(min_val-max_val)
  min_val, max_val = min_val - delta*0.10, max_val + delta*0.10
  p = figure(width=400, height=400, x_range=(min_val, max_val), y_range=(min_val, max_val), match_aspect=True,aspect_scale=1)
  p.title = "Persistence diagram"
  p.xaxis.axis_label = "Birth"
  p.yaxis.axis_label = "Death"
  x = dgm["birth"][dgm["death"] != np.inf]
  y = dgm["death"][dgm["death"] != np.inf]
  p.scatter(x,y, size=pt_size)
  x = dgm["birth"][dgm["death"] == np.inf]
  y = np.repeat(max_val - delta*0.05, sum(dgm["death"] == np.inf))
  s = Span(dimension="width", location=max_val - delta*0.05, line_width=1.0, line_color="gray", line_dash="dotted")
  s.level = 'underlay'
  p.add_layout(s)
  p.scatter(x,y, size=pt_size, color="red")
  p.patch([min_val, max_val, max_val], [min_val, min_val, max_val], line_width=0, fill_color="gray", fill_alpha=0.80)
  show(p)
  return p

def plot_dist(d: Sequence[float], palette: str = "viridis", **kwargs):
  n = inverse_choose(len(d), 2)
  C = np.floor(bin_color(d)*255).astype(int)
  D = np.zeros((n,n), dtype=np.uint32)
  D_view = D.view(dtype=np.int8).reshape((n, n, 4))
  for cc, (i,j) in enumerate(combinations(range(n),2)):
    D_view[i,j,0] = D_view[j,i,0] = C[cc,0]
    D_view[i,j,1] = D_view[j,i,1] = C[cc,1]
    D_view[i,j,2] = D_view[j,i,2] = C[cc,2]
    D_view[i,j,3] = D_view[j,i,3] = 255
  min_col = (bin_color([0, 1])[0,:]*255).astype(int)
  for i in range(n):
    D_view[i,i,0] = 68
    D_view[i,i,1] = 2
    D_view[i,i,2] = 85
    D_view[i,i,3] = 255
  fig_kw = dict(width=200, height=200) | kwargs
  p = figure(**fig_kw)
  p.x_range.range_padding = p.y_range.range_padding = 0
  # p.y_range.flipped = True
  # p.y_range = Range1d(0,-10)
  # p.x_range = Range1d(0,10)
  p.image_rgba(image=[np.flipud(D)], x=0, y=0, dw=10, dh=10)
  show(p)


def plot_complex(S: SimplicialComplex, pos: ArrayLike = None, color: Optional[ArrayLike] = None, palette: str = "viridis", notebook: bool = True, bin_kwargs = None, **kwargs):
  """
  Plots a simplicial complex in 2D with Bokeh 
  
  Parameters: 
    S := simplicial complex
    pos := one of 'mds', 'spring', or a np.ndarray instance of vertex positions
    notebook := use Bokeh's 'output_notebook()' functionality
    **kwargs := additional arguments to pass to figure()
  """
  
  ## Default scales
  if (notebook): output_notebook(verbose=False, hide_banner=True)
  TOOLTIPS = [ ("index", "$value") ]

  color = np.ones(len(S)) if color is None else color
  #assert isinstance(color, Sequence)
  if len(color) == S.shape[0]:
    v_color = color
    e_color = [max(color[e]) for e in faces(S,1)]
    t_color = [max(color[t]) for t in faces(S,2)]
  elif len(color) == len(S):
    d = np.array([len(s)-1 for s in faces(S)], dtype=np.int8)
    v_color = color[d==0]
    e_color = color[d==1]
    t_color = color[d==2]
  else: 
    raise ValueError(f"Invalid color argument {type(color)}")


  ## Deduce embedding
  from scipy.sparse import diags
  from pbsig.linalg import adjacency_matrix
  pos = "mds" if pos is None else pos
  use_grid_lines = False
  A = adjacency_matrix(S)
  if isinstance(pos, str) and pos == "spring":
    import networkx as nx
    G = nx.from_numpy_array(A)
    pos = np.array(list(nx.spring_layout(G).values()))
  elif isinstance(pos, str) and pos == "mds":
    from scipy.sparse.csgraph import floyd_warshall
    pos = cmds(floyd_warshall(A, directed=False, unweighted=True)**2)
    use_grid_lines = True
  elif isinstance(pos, np.ndarray) and pos.shape[1] == 2:
    assert pos.shape[0] == S.shape[0], "Layout must match number of vertices"
    use_grid_lines = True
  else:
    raise ValueError("Unimplemented layout")

  ## Create a plot — set dimensions, toolbar, and title
  x_rng = np.array([np.min(pos[:,0]), np.max(pos[:,0])])*[0.90, 1.10]
  y_rng = np.array([np.min(pos[:,1]), np.max(pos[:,1])])*[0.90, 1.10]
  p = figure(
    tools="pan,wheel_zoom,reset", 
    active_scroll=None,
    active_drag="auto",
    # x_range=x_rng, 
    # y_range=y_rng,  
    tooltips=TOOLTIPS,
    plot_width=300, 
    plot_height=300,
    **kwargs
  )
  p.axis.visible = False
  p.xgrid.visible = use_grid_lines
  p.ygrid.visible = use_grid_lines

  default_bin_kwargs = dict(lb=min(color), ub=max(color), color_pal=palette) 
  bin_kwargs = default_bin_kwargs if bin_kwargs is None else (default_bin_kwargs | bin_kwargs)

  ## Create the (p >= 2)-simplex renderer
  if len(S.shape) > 2 and S.shape[2] > 0:
    t_x = [pos[[i,j,k],0] for (i,j,k) in S.faces(2)]
    t_y = [pos[[i,j,k],1] for (i,j,k) in S.faces(2)]
    t_col = bin_color(t_color, **bin_kwargs)
    t_data = {
      'xs' : t_x,
      'ys' : t_y,
      'color' : t_col, # np.repeat("#808080", len(edge_x)),
      'value' : t_color
    }
    t_source = ColumnDataSource(data=t_data)
    t_renderer = p.patches('xs', 'ys', color='color', alpha=0.70, line_width=2, source=t_source)

  ## Create edge renderer
  if len(S.shape) > 1 and S.shape[1] > 0:
    e_scale = 0.25
    e_x = [pos[e,0] for e in S.faces(1)]
    e_y = [pos[e,1] for e in S.faces(1)]
    e_sizes = np.ones(S.shape[1]) #np.array(e_sizes)
    e_widths = (e_sizes / np.max(e_sizes))*e_scale
    e_col = bin_color(e_color,  **bin_kwargs)
    #ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)
    e_data = {
      'xs' : e_x,
      'ys' : e_y,
      'color' : e_col, # np.repeat("#808080", len(edge_x)),
      'line_width': e_widths, 
      'value' : e_color
    }
    e_source = ColumnDataSource(data=e_data)
    e_renderer = p.multi_line('xs', 'ys', color='color', line_width='line_width', alpha=1.00, source=e_source)
    
  ## Create node renderer
  if len(S.shape) > 0 and S.shape[0] > 0:
    v_scale = 35.0
    v_col = bin_color(v_color,  **bin_kwargs)
    v_data = {
      'x' : pos[:,0],
      'y' : pos[:,1],
      'size' : np.repeat(v_scale, S.shape[0]), 
      'color' : v_col,
      'value' : v_color
    }
    v_source = ColumnDataSource(data=v_data)
    v_renderer = p.circle(x='x', y='y', color='color', alpha=1.0, source=v_source)

  p.toolbar.logo = None
  show(p)
