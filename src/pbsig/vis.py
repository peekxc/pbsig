
import numpy as np
from itertools import * 
import networkx as nx
from typing import * 
from numbers import Number

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

## Meta-programming to the rescue! 
def valid_parameters(el: Any, prefix: str = "", exclude: list = [], **kwargs):
  """Extracts valid parameters for a bokeh plotting element. 
  
  This function takes as input a bokeh model (i.e. figure, Scatter, MultiLine, Patch, etc.) and a set of keyword arguments prefixed by 'prefix', 
  and extracts from the the given keywords a dictionary of valid parameters for the given element, with the chosen prefix removed. 

  Example: 

    >>> valid_parameters(MultiLine, prefix="multiline_", multiline_linewidth=2.0)
    # { 'linewidth' : 2.0 }

  Parameters: 
    el: bokeh model or glyph element with a .parameters() function
    prefix: prefix to extract. Defaults to empty string. 
    kwargs: keyword arguments to extract the valid parameters from.

  """
  assert hasattr(el, "parameters"), f"Invalid bokeh element '{type(el)}'; must have valid parameters() call"
  param_names = { param[0].name for param in el.parameters() }
  stripped_params = { k[len(prefix):] for k in kwargs.keys() if k.startswith(prefix) }
  valid_params = { p : kwargs[prefix+p] for p in (stripped_params & param_names) if not p in exclude }
  return valid_params

# def intersect_dict(d1: dict, d2: dict):
#   intersect_keys = set(d1.keys() & d2.keys())
#   { d1}

def figure_dgm(dgm: ArrayLike = None, pt_size: int = 5, show_filter: bool = False, **kwargs):
  from pbsig.persistence import as_dgm
  default_figkwargs = dict(width=300, height=300, match_aspect=True, aspect_scale=1, title="Persistence diagram")
  fig_kwargs = default_figkwargs.copy()
  if dgm is None or len(dgm) == 0:
    fig_kwargs["x_range"] = (0, 1)
    fig_kwargs["y_range"] = (0, 1)
    min_val = 0
    max_val = 1
  else: 
    dgm = as_dgm(dgm)
    max_val = max(np.ravel(dgm["death"]), key=lambda v: v if v != np.inf else -v) 
    max_val = (max_val if max_val != np.inf else max(dgm["birth"])*5)
    min_val = min(np.ravel(dgm["birth"]))
    min_val = (min_val if min_val != max_val else 0.0)
    delta = abs(min_val-max_val)
    min_val, max_val = min_val - delta*0.10, max_val + delta*0.10
    fig_kwargs["x_range"] = (min_val, max_val)
    fig_kwargs["y_range"] = (min_val, max_val)

  ## Parameterize the figure
  from bokeh.models import PolyAnnotation
  fig_kwargs = valid_parameters(figure, **(fig_kwargs | kwargs))
  p = kwargs.get('figure', figure(**fig_kwargs))
  p.xaxis.axis_label = "Birth"
  p.yaxis.axis_label = "Death"
  polygon = PolyAnnotation(
    fill_color="gray", fill_alpha=1.0,
    xs=[min_val-100, max_val+100, max_val+100],
    ys=[min_val-100, min_val-100, max_val+100],
    line_width=0
  )
  p.add_layout(polygon)
  # p.patch([min_val-100, max_val+100, max_val+100], [min_val-100, min_val-100, max_val+100], line_width=0, fill_color="gray", fill_alpha=0.80)
  
  ## Plot non-essential points, where applicable 
  if dgm is not None and any(dgm["death"] != np.inf):
    x = dgm["birth"][dgm["death"] != np.inf]
    y = dgm["death"][dgm["death"] != np.inf]
    p.scatter(x,y, size=pt_size)

  ## Plot essential points, where applicable 
  if dgm is not None and any(dgm["death"] == np.inf):
    x = dgm["birth"][dgm["death"] == np.inf]
    y = np.repeat(max_val - delta*0.05, sum(dgm["death"] == np.inf))
    s = Span(dimension="width", location=max_val - delta*0.05, line_width=1.0, line_color="gray", line_dash="dotted")
    s.level = 'underlay'
    p.add_layout(s)
    p.scatter(x,y, size=pt_size, color="red")

  return p

def plot_dgm(*args, **kwargs) -> None:
  show(figure_dgm(*args, **kwargs))

def figure_dist(d: Sequence[float], palette: str = "viridis", **kwargs):
  n = inverse_choose(len(d), 2)
  C = np.floor(bin_color(np.append([0], d), palette)*255).astype(int)
  base_color, C = C[0], C[1:]
  D = np.zeros((n,n), dtype=np.uint32)
  D_view = D.view(dtype=np.int8).reshape((n, n, 4))
  for cc, (i,j) in enumerate(combinations(range(n),2)):
    D_view[i,j,0] = D_view[j,i,0] = C[cc,0]
    D_view[i,j,1] = D_view[j,i,1] = C[cc,1]
    D_view[i,j,2] = D_view[j,i,2] = C[cc,2]
    D_view[i,j,3] = D_view[j,i,3] = 255
  # min_col = (bin_color([0, 1])[0,:]*255).astype(int)
  for i in range(n):
    D_view[i,i,0] = base_color[0]
    D_view[i,i,1] = base_color[1]
    D_view[i,i,2] = base_color[2]
    D_view[i,i,3] = 255
  fig_kw = valid_parameters(figure, **(dict(width=300, height=300) | kwargs))
  p = kwargs.get('figure', figure(**fig_kw))
  p.x_range.range_padding = p.y_range.range_padding = 0
  # p.y_range.flipped = True
  # p.y_range = Range1d(0,-10)
  # p.x_range = Range1d(0,10)
  # D_view[0,0,0] = 255
  # D_view[0,0,1] = 0
  # D_view[0,0,2] = 0
  # D_view[0,0,3] = 255
  # D_view[1,1,0] = 0
  # D_view[1,1,1] = 255
  # D_view[1,1,2] = 0
  # D_view[1,1,3] = 255
  p.image_rgba(image=[np.flipud(D)], x=0, y=0, dw=10, dh=10)
  return p

def plot_dist(*args, **kwargs) -> None:
  show(dist_figure(*args, **kwargs))

def figure_point_cloud(X: ArrayLike, **kwargs):
  p = kwargs.get('figure', figure(**kwargs))
  p.scatter(X[:,0], X[:,1])
  return p


def figure_complex(
  S: ComplexLike, 
  pos: ArrayLike = None, 
  color: Optional[ArrayLike] = None, 
  palette: str = "viridis", 
  bin_kwargs = None, 
  **kwargs
):
  """
  Plots a simplicial complex in 2D with Bokeh. 
  
  Parameters: 
    S: _ComplexLike_ representing a simplicial complex.
    pos: where to position the vertices. Can be one of 'mds', 'spring', or a ndarray instance of vertex positions.
    color: either a 2-d ndarray of colors, or a 1-d array of weights to infer colors from via _palette_
    palette: color palette (string) indicating which color palette to use. 
    bin_kwargs: arguments to pass to _bin\_color()_ when _color_ is supplied
    simplex_kwargs: 
    kwargs := additional arguments to pass to figure()
  """
  TOOLTIPS = [ ("index", "$value") ]
  # simplex_kwargs = simplex_kwargs if simplex_kwargs is not None else {}

  ## Default color values == dimension of the simplex 
  d = np.array([len(s)-1 for s in faces(S)], dtype=np.int8)
  if color is None:
    cv = d.copy()
  elif len(color) == len(S):
    cv = np.array(color)
  elif len(color) == card(S, 0):
    cv = np.zeros(len(S))
    cv[d == 0] = color
    for di in range(1, dim(S)+1):
      cv[d == di] = [max(color[e]) for e in faces(S,di)]
  else: 
    raise ValueError(f"Invalid color argument {type(color)}")
  assert isinstance(cv, Container)

  ## Replace color with rgba values from color palette 
  if cv.ndim == 1:
    default_bin_kwargs = dict(lb=min(cv), ub=max(cv), color_pal=palette) 
    bin_kwargs = default_bin_kwargs if bin_kwargs is None else (default_bin_kwargs | bin_kwargs)
    color = bin_color(cv, **bin_kwargs)
  else:
    color = cv

  ## Deduce embedding
  from scipy.sparse import diags
  from pbsig.csgraph import adjacency_matrix
  pos = "mds" if pos is None else pos
  use_grid_lines = False
  if isinstance(pos, str) and pos == "spring":
    import networkx as nx
    A = adjacency_matrix(S)
    G = nx.from_numpy_array(A)
    pos = np.array(list(nx.spring_layout(G).values()))
  elif isinstance(pos, str) and pos == "mds":
    from scipy.sparse.csgraph import floyd_warshall
    A = adjacency_matrix(S)
    pos = cmds(floyd_warshall(A, directed=False, unweighted=True)**2)
    use_grid_lines = True
  elif isinstance(pos, np.ndarray) and pos.shape[1] == 2:
    assert pos.shape[0] == card(S, 0), "Layout must match number of vertices"
    use_grid_lines = True
  else:
    raise ValueError("Unimplemented layout")

  ## Create a plot — set dimensions, toolbar, and title
  fig_params = valid_parameters(figure, **kwargs)
  x_rng = np.array([np.min(pos[:,0]), np.max(pos[:,0])])*[0.90, 1.10]
  y_rng = np.array([np.min(pos[:,1]), np.max(pos[:,1])])*[0.90, 1.10]
  fig_params |= dict(tools="pan,wheel_zoom,reset", active_scroll=None, active_drag="auto", tooltips=TOOLTIPS)
  p = kwargs.get('figure', figure(**fig_params))
  p.axis.visible = use_grid_lines
  p.xgrid.visible = use_grid_lines
  p.ygrid.visible = use_grid_lines


  ## Create the (p == 2)-simplex renderer
  if card(S, 2) > 0:
    from bokeh.models import Patch
    t_x = [pos[[i,j,k],0] for (i,j,k) in faces(S, 2)]
    t_y = [pos[[i,j,k],1] for (i,j,k) in faces(S, 2)]
    t_data = {
      'xs' : t_x,
      'ys' : t_y,
      'color' : color[d == 2] # np.repeat("#808080", len(edge_x)),
      # 'value' : t_color
    }
    t_source = ColumnDataSource(data=t_data)
    patch_params = valid_parameters(Patch, prefix="patch_", **kwargs)
    patch_params.update(valid_parameters(Patch, prefix="triangle_", **kwargs))
    patch_params['alpha'] = patch_params.get('alpha', 0.20)
    patch_params['line_width'] = patch_params.get('line_width', 2)
    t_renderer = p.patches('xs', 'ys', color='color', source=t_source, **patch_params)

  ## Create edge renderer
  if card(S, 1) > 0:
    from bokeh.models import MultiLine
    edge_params = valid_parameters(MultiLine, prefix="multiline_", **kwargs)
    edge_params.update(valid_parameters(MultiLine, prefix="edge_", **kwargs))
    e_line_width = edge_params.pop('line_width', 0.75)
    e_alpha = edge_params.pop('line_alpha', 1.0)
    e_x = [pos[e,0] for e in faces(S, 1)]
    e_y = [pos[e,1] for e in faces(S, 1)]
    #ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)
    e_data = {
      'xs' : e_x,
      'ys' : e_y,
      'color' : color[d == 1], # np.repeat("#808080", len(edge_x)),
      'line_width': np.repeat(e_line_width, card(S,1)) if isinstance(e_line_width, Number) else e_line_width, 
      'alpha' : np.repeat(e_alpha, card(S,1)) if isinstance(e_alpha, Number) else e_alpha, 
    }
    e_source = ColumnDataSource(data=e_data)
    e_renderer = p.multi_line('xs', 'ys', color='color', line_width='line_width', source=e_source, **edge_params)
    
  ## Create node renderer
  if card(S, 0) > 0:
    from bokeh.models import Circle
    vertex_params = valid_parameters(Circle, prefix="circle_", **kwargs)
    vertex_params.update(valid_parameters(Circle, prefix="vertex_", **kwargs))
    v_alpha = vertex_params.pop('alpha', 1.0)
    v_size = vertex_params.pop('size', 6.0)
    v_data = {
      'x' : pos[:,0],
      'y' : pos[:,1],
      'size' : np.repeat(v_size, card(S, 0)) if isinstance(v_size, Number) else v_size, 
      'alpha' : np.repeat(v_alpha, card(S, 0)) if isinstance(v_alpha, Number) else v_alpha,
      'color' : color[d == 0]
    }
    v_source = ColumnDataSource(data=v_data)
    v_renderer = p.circle(x='x', y='y', color='color', size='size', source=v_source, **vertex_params)
  
  p.toolbar.logo = None
  return p 

def plot_complex(*args, **kwargs) -> None:
  show(figure_complex(*args, **kwargs))

def dict_setdiff(d1: dict, d2: dict):
  keys_to_keep = set(d1.keys()) - set(d2.keys())
  return { k : d1[k] for k in keys_to_keep }

def figure_scatter(X: ArrayLike, **kwargs):
  from bokeh.models import Scatter
  # fig_params = { param[0].name for param in figure.parameters() }
  # # scatter_params = { param[0].name for param in Scatter.parameters() }
  # p = kwargs.get('figure', figure(**({ k : kwargs[k] for k in kwargs.keys() & fig_params })))
  # p.scatter(*X.T, **({ k : kwargs[k] for k in kwargs.keys() - fig_params }))
  fig_kwargs = valid_parameters(figure, kwargs)
  p = kwargs.get('figure', figure(**fig_kwargs))
  Scatter_kwargs = valid_parameters(Scatter, **kwargs)
  Scatter_kwargs['x'] = X[:,0]
  Scatter_kwargs['y'] = X[:,1]
  cs_kwargs = { k : v for k,v in Scatter_kwargs.items() if isinstance(v, Sized) and len(v) == len(X) }
  source = ColumnDataSource(cs_kwargs)
  Scatter_kwargs = dict_setdiff(Scatter_kwargs, cs_kwargs)
  for k in cs_kwargs.keys():
    Scatter_kwargs[k] = str(k)
  glyph = Scatter(**Scatter_kwargs)
  p.add_glyph(source, glyph)
  # p.scatter(*X.T, **scatter_kwargs)
  return p

def figure_patch(X: ArrayLike, **kwargs):
  fig_params = { param[0].name for param in figure.parameters() }
  # patch_params = { param[0].name for param in Scatter.parameters() }
  p = kwargs.get('figure', figure(**({ k : kwargs[k] for k in kwargs.keys() & fig_params })))
  p.patch(*X.T, **({ k : kwargs[k] for k in kwargs.keys() - fig_params }))
  return p

def figure_hist(hist, edges, **kwargs):
  from bokeh.palettes import HighContrast3
  blue, orange, red = HighContrast3[0], HighContrast3[1], HighContrast3[2]
  p = kwargs.get('figure', figure(tools='', background_fill_color="#fafafa"))
  p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=blue, line_color="white", alpha=0.5)
  p.y_range.start = 0
  p.xaxis.axis_label = 'x'
  p.yaxis.axis_label = 'Pr(x)'
  p.grid.grid_line_color="white"
  return p

def figure_vineyard(dgms: Sequence[dict], p: int = None, **kwargs):
  from bokeh.models import Range1d
  from numbers import Integral
  from pbsig.color import rgb_to_hex
  fig = figure_dgm(**kwargs)
  vine_colors = bin_color(np.arange(len(dgms)), "viridis")
  vine_colors = (vine_colors*255).astype(np.uint8)
  p = [p] if isinstance(p, Integral) else p
  assert isinstance(p, Iterable), "dimension 'p' must be a iterable or an integer."
  p_max = max(p)
  min_birth = np.min([np.min(d[0]['birth']) for d in dgms])
  max_death = np.max([np.max(np.where(np.isinf(d[p_max]['death']), 0.0, d[p_max]['death'])) for d in dgms if p_max in d])
  lifetime = (max_death - min_birth)
  fig.x_range = Range1d(min_birth - lifetime*0.05, max_death + lifetime*0.05)
  fig.y_range = Range1d(min_birth - lifetime*0.05, max_death + lifetime*0.05)
  for dgm, vc in zip(dgms, vine_colors):
    for p_dim in p: 
      if p_dim in dgm:
        fig.scatter(dgm[p_dim]['birth'], dgm[p_dim]['death'], color=rgb_to_hex(vc))
  fig.match_aspect = True
  return fig


def figure_plain(p: figure):
  """Turns off the visibility of the toolbar, grid axis, and background lines of a given figure."""
  assert isinstance(p, figure), "Supplied object is not a figure object!"
  p.toolbar_location = None
  p.xaxis.visible = False
  p.yaxis.visible = False
  p.xgrid.visible = False
  p.ygrid.visible = False
  p.min_border_left = 0
  p.min_border_right = 0
  p.min_border_top = 0
  p.min_border_bottom = 0
  return p

def g_edges(p: figure, edges: Iterable, pos: ArrayLike, **kwargs):
  e_scale = 0.75
  e_x = [pos[e,0] for e in edges]
  e_y = [pos[e,1] for e in edges]
  e_sizes = np.ones(len(edges))
  e_widths = (e_sizes / np.max(e_sizes))*e_scale
  #ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)
  e_data = {
    'xs' : e_x,
    'ys' : e_y,
    'line_width': e_widths
  } | kwargs
  # e_source = ColumnDataSource(data=e_data)
  e_renderer = p.multi_line(**e_data)
  # e_renderer = p.multi_line('xs', 'ys', color='color', line_width='line_width', alpha=1.00, source=e_source)
  return p