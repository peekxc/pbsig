import numpy as np
from scipy.sparse import csc_array, coo_array

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
  A = coo_array(A).astype(float)
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

from pbsig.simplicial import *
def is_connected(S: ComplexLike):
  from scipy.cluster.hierarchy import DisjointSet
  ds = DisjointSet(list(S.faces(0))) 
  for i,j in faces(S, 1):
    ds.merge(Simplex(i), Simplex(j))
  return ds.n_subsets == 1

def connected_components(S: ComplexLike):
  from scipy.cluster.hierarchy import DisjointSet
  ds = DisjointSet(list(S.faces(0))) 
  for i,j in faces(S, 1):
    ds.merge(Simplex(i), Simplex(j))
  p = np.zeros(card(S,0), dtype=int)
  for i, subset in enumerate(ds.subsets()):
    for representative in subset:
      p[int(representative[0])] = i
  return p 

def sparsify(A, epsilon: float = 1.0, ensure_connected: bool = False, max_tries = 10, exclude_singletons: bool = True):
  """
  Sparsifies adjacency matrix 'A' 
  """
  jl, Main = import_julia()
  A = coo_array(A).astype(float)
  Main.I = np.array(A.row+1).astype(float)
  Main.J = np.array(A.col+1).astype(float)
  Main.V = A.data # abs(A.data) if weighted else np.repeat(1, len(A.data))
  jl.eval("S = sparse(I,J,V); 0")
  
  ## Disable warnings
  Main.using("Logging")
  jl.eval("Logging.disable_logging(Logging.Warn);")

  ## Different notions of connectdness 
  def _is_connected(exclude_singletons: bool = False):
    if not exclude_singletons:
      SA = csc_array(jl.eval("SA"))
      return is_connected(simplicial_complex(chain(range(A.shape[0]), zip(*SA.nonzero()))))
    else: 
      return jl.eval("isConnected(SA)")

  jl.eval(f"SA = sparsify(S, ep={epsilon}); 0")
  if ensure_connected:
    n_tries = 1
    while n_tries < max_tries and not _is_connected(exclude_singletons):
      jl.eval(f"SA = sparsify(S, ep={epsilon}); 0")
      n_tries += 1
    # aq = jl.eval("approxQual(S, SA)")
  if n_tries == max_tries:
    raise RuntimeError("Unable to ensure the graph stays connected")
  SA = csc_array(jl.eval("SA"))
  return SA

