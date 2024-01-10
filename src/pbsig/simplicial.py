import numpy as np
import splex as sx
from .utility import lexsort_rows, chain, cycle_window

def cycle_graph(n: int):
  "Creates a cycle graph from a set of ordered points" 
  E = np.array(list(cycle_window(range(n))))
  V = np.fromiter(range(n), dtype=np.int32)
  S = sx.simplicial_complex(chain(iter(V), iter(lexsort_rows(E))))
  return S

def complete_graph(n: int):
  "Creates the complete graph from an integer n" 
  from itertools import combinations
  V = np.fromiter(range(n), dtype=np.int32)
  E = np.array(list(combinations(range(n), 2)), dtype=np.int32)
  S = sx.simplicial_complex(chain(iter(V), iter(lexsort_rows(E))))
  return(S)

def path_graph(n: int):
  from more_itertools import pairwise
  V = np.fromiter(range(n), dtype=np.int32)
  E = np.array(list(pairwise(V)), dtype=np.int32)
  S = sx.simplicial_complex(chain(iter(V), iter(lexsort_rows(E))))
  return S


def simplify_complex(S: sx.ComplexLike, pos: np.ndarray, **kwargs):
  # import open3d as o3
  # dir(o3)
  # o3.io.write_triangle_mesh?
  # from pbsig.linalg import pca
  from fast_simplification import simplify
  default_kwargs = dict(target_reduction = 0.90, agg = 1) | kwargs
  triangles = np.array(sx.faces(S, 2))
  pos = pos.astype(np.float32) if pos.dtype != np.float32 else pos
  pos_new, tri_new = simplify(pos, triangles, **default_kwargs)
  S_new = sx.simplicial_complex(tri_new, form='rank')
  return S_new, pos_new
  
  