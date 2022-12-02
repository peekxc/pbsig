import bokeh 
import networkx as nx

def plot_graph(G):
  from bokeh.plotting import figure
  from bokeh.models import GraphRenderer, Ellipse
  from bokeh.palettes import Spectral8

  # list the nodes and initialize a plot
  N = G.number_of_nodes()
  node_indices = list(range(N))
  plot = figure(title="Graph layout demonstration", tools="", toolbar_location=None)
  graph = GraphRenderer()

  graph.node_renderer.glyph = Ellipse(height=0.1, width=0.2, fill_color="fill_color")
  graph.node_renderer.data_source.data = dict(index=node_indices, fill_color=Spectral8)
  graph.edge_renderer.data_source.data = dict(start=[0]*N, end=node_indices)
