import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

def low_stretch_st(G, method: str):
  import networkx as nx
  assert isinstance(G, nx.Graph)
  from julia.api import Julia
  jl = Julia(compiled_modules=False) # it's unclear why this is needed, but it is
  from julia import Main
  Main.using("Laplacians")
  Main.using("LinearAlgebra")
  Main.using("SparseArrays")
  # G = nx.connected_watts_strogatz_graph(10, k=3, p=0.20)
  A = coo_matrix(nx.adjacency_matrix(G)).astype(float)
  Main.I = np.array(A.row+1).astype(float)
  Main.J = np.array(A.col+1).astype(float)
  Main.V = A.data 
  jl.eval("S = sparse(I,J,V)")
  if method == "akpw":
    st = np.array(jl.eval("t = akpw(S)"))
  elif method == "randishPrim":
    st = np.array(jl.eval("t = akpw(S)"))
  elif method == "randishKruskal":
    st = np.array(jl.eval("t = randishKruskal(S)"))
  else:
    raise ValueError("invalid method")
  return st

