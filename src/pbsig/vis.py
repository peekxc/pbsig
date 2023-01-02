import bokeh 
import networkx as nx
import numpy as np

from pbsig.linalg import * # cmds

from bokeh.plotting import figure, show
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import GraphRenderer, Ellipse, Range1d, Circle, ColumnDataSource, MultiLine, Label, LabelSet, Button
from bokeh.palettes import Spectral8, RdGy
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.io import output_notebook, show, save
from bokeh.transform import linear_cmap
from bokeh.layouts import column


def plot_complex(S, pos = ["spring", "mds"], notebook=True, **kwargs):
  
  ## Default scales
  vertex_scale, edge_scale = 6.0, 0.25 
  if (notebook): output_notebook(verbose=False, hide_banner=True)
  TOOLTIPS = [ ("index", "$index") ]

  ## Deduce embedding
  pos = "spring" if pos == ["spring", "mds"] else pos
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
  else:
    raise ValueError("Unimplemented layout")

  ## Create a plot — set dimensions, toolbar, and title
  x_rng = np.array([np.min(pos[:,0]), np.max(pos[:,0])])*[0.90, 1.10]
  y_rng = np.array([np.min(pos[:,1]), np.max(pos[:,1])])*[0.90, 1.10]
  p = figure(
    tools="pan,wheel_zoom,reset", 
    active_scroll=None,
    active_drag="auto",
    x_range=x_rng, 
    y_range=y_rng, 
    title=S.__repr__(), 
    tooltips=TOOLTIPS,
    plot_width=300, 
    plot_height=300,
    **kwargs
  )
  p.axis.visible = False
  p.xgrid.visible = use_grid_lines
  p.ygrid.visible = use_grid_lines
  edge_x = [pos[e,0] for e in G.edges]
  edge_y = [pos[e,1] for e in G.edges]

  ## Edge widths
  from pbsig.color import bin_color, colors_to_hex, linear_gradient
  e_sizes = np.ones(S.shape[1]) #np.array(e_sizes)
  e_widths = (e_sizes / np.max(e_sizes))*edge_scale
  #ec = bin_color(ec, linear_gradient(["gray", "red"], 100)['hex'], min_x = 0.0, max_x=1.0)

  ## Create edge renderer
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

# def plot_graph(G):
#   from bokeh.plotting import figure
#   from bokeh.models import GraphRenderer, Ellipse
#   from bokeh.palettes import Spectral8

#   # list the nodes and initialize a plot
#   N = G.number_of_nodes()
#   node_indices = list(range(N))
#   plot = figure(title="Graph layout demonstration", tools="", toolbar_location=None)
#   graph = GraphRenderer()

#   graph.node_renderer.glyph = Ellipse(height=0.1, width=0.2, fill_color="fill_color")
#   graph.node_renderer.data_source.data = dict(index=node_indices, fill_color=Spectral8)
#   graph.edge_renderer.data_source.data = dict(start=[0]*N, end=node_indices)
