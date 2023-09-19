# %% [markdown] 
# ---
# title: Mesh Simplification with a spectral rank invariant
# author: Matt Piekenbrock
# date: 09/14/23
# ---

# %% Imports
#| echo: false
import numpy as np
import scipy as sp
from itertools import *
from typing import * 
import networkx as nx
import numpy as np


# # import open3d as o3
# from more_itertools import chunked, pairwise
# from pbsig.betti import Sieve
# from scipy.sparse.csgraph import floyd_warshall
# from splex import *
# import bokeh
# from bokeh.plotting import show, figure 
# from bokeh.models import Range1d
# from bokeh.layouts import row, column
# from bokeh.io import output_notebook
# output_notebook()

# %% [markdown]
### Loading the pose mesh data 
# %%
# | label: Loading the pose mesh data 
# | echo: true
from splex import *
from pbsig.datasets import pose_meshes
mesh_loader = pose_meshes(simplify=1000, which=["elephant"]) # 8k triangles takes ~ 40 seconds


# %% [markdown]
# Geodesic calculation with open3d meshes using floyd warshall

# %% [python]
#| echo: true
def geodesics(mesh):
  from scipy.sparse.csgraph import floyd_warshall
  from scipy.spatial.distance import squareform, cdist
  import networkx as nx
  vertices = np.asarray(mesh.vertices)
  mesh.compute_adjacency_list()
  G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(mesh.adjacency_list) })
  A = nx.adjacency_matrix(G).tocoo()
  A.data = np.linalg.norm(vertices[A.row] - vertices[A.col], axis=1)
  #A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
  AG = floyd_warshall(A.tocsr())
  return AG

# %% 


