
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
from bokeh.models import GraphRenderer, Ellipse, Range1d, Circle, ColumnDataSource, MultiLine, Label, LabelSet, Button
from bokeh.palettes import Spectral8, RdGy
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.io import output_notebook, show, save
from bokeh.transform import linear_cmap
from bokeh.layouts import column


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


def plot_complex(S, pos = None, notebook=True, **kwargs):
  """
  Plots a simplicial complex in 2D with Bokeh 
  
  Parameters: 
    S := simplicial complex
    pos := one of 'mds', 'spring', or a np.ndarray instance of vertex positions
    notebook := use Bokeh's 'output_notebook()' functionality
    **kwargs := additional arguments to pass to figure()
  """
  
  ## Default scales
  vertex_scale, edge_scale = 6.0, 0.25 
  if (notebook): output_notebook(verbose=False, hide_banner=True)
  TOOLTIPS = [ ("index", "$index") ]

  ## Deduce embedding
  pos = "mds" if pos is None else pos
  use_grid_lines = False
  L = up_laplacian(S, p = 0)
  A = L - diags(L.diagonal())
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


  ## Create the (p >= 2)-simplex renderer
  

  ## Create edge renderer
  edge_x = [pos[e,0] for e in S.faces(1)]
  edge_y = [pos[e,1] for e in S.faces(1)]
  e_sizes = np.ones(S.shape[1]) #np.array(e_sizes)
  e_widths = (e_sizes / np.max(e_sizes))*edge_scale
  #ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)
  edge_data = {
    'xs' : edge_x,
    'ys' : edge_y,
    'color' : np.repeat("#808080", len(edge_x)),
    'line_width': e_widths
  }
  edge_source = ColumnDataSource(data=edge_data)
  edge_renderer = p.multi_line('xs', 'ys', color='color', line_width='line_width', alpha=0.80, source=edge_source)
  
  ## Create node renderer
  node_data = {
    'x' : pos[:,0],
    'y' : pos[:,1],
    'size' : np.repeat(vertex_scale, S.shape[0])
  }
  node_source = ColumnDataSource(data=node_data)
  node_renderer = p.circle('x', 'y', color="red", alpha=1.0, source=node_source)

  p.toolbar.logo = None
  show(p)
