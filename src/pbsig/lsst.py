import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

# sparsify(nx.adjacency_matrix(G))

def import_julia():
  from julia.api import Julia
  jl = Julia(compiled_modules=False) # it's unclear why this is needed, but it is
  from julia import Main
  Main.using("Laplacians")
  Main.using("LinearAlgebra")
  Main.using("SparseArrays")
  return jl, Main

def low_stretch_st(A, method: str = "akpw", weighted: bool = True):
  """ Given a (possibly edge-weighted) networkx graph 'G', find a low-stretch spanning tree T of G """
  # import networkx as nx
  # assert isinstance(G, nx.Graph)
  jl, Main = import_julia()
  A = coo_matrix(A).astype(float)
  Main.I = np.array(A.row+1).astype(float)
  Main.J = np.array(A.col+1).astype(float)
  Main.V = abs(A.data) if weighted else np.repeat(1, len(A.data))
  st = jl.eval("S = sparse(I,J,V); 0")
  if method == "akpw":
    st = np.array(jl.eval("t = akpw(S)"))
  elif method == "randishPrim":
    st = np.array(jl.eval("t = akpw(S)"))
  elif method == "randishKruskal":
    st = np.array(jl.eval("t = randishKruskal(S)"))
  else:
    raise ValueError("invalid method")
  return st

def sparsify(A, epsilon: float = 1.0, ensure_connected: bool = False, max_tries = 10):
  jl, Main = import_julia()
  A = coo_matrix(A).astype(float)
  Main.I = np.array(A.row+1).astype(float)
  Main.J = np.array(A.col+1).astype(float)
  Main.V = A.data # abs(A.data) if weighted else np.repeat(1, len(A.data))
  jl.eval("S = sparse(I,J,V); 0")
  jl.eval(f"SA = sparsify(S, ep={epsilon}); 0")
  if ensure_connected:
    n_tries = 1
    while n_tries < 10 and not jl.eval("isConnected(SA)"):
      jl.eval(f"SA = sparsify(S, ep={epsilon}); 0")
      n_tries += 1
    # aq = jl.eval("approxQual(S, SA)")
  SA = csc_matrix(jl.eval("SA"))
  return SA

