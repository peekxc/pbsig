from splex import * 
from .utility import *

def cycle_graph(X: ArrayLike):
  "Creates a cycle graph from a set of ordered points" 
  E = np.array(list(cycle_window(range(X.shape[0]))))
  V = np.fromiter(range(X.shape[0]), dtype=np.int32)
  S = simplicial_complex(chain(iter(V), iter(lexsort_rows(E))))
  return S

def complete_graph(n: int):
  "Creates the complete graph from an integer n" 
  from itertools import combinations
  V = np.fromiter(range(n), dtype=np.int32)
  E = np.array(list(combinations(range(n), 2)), dtype=np.int32)
  S = simplicial_complex(chain(iter(V), iter(lexsort_rows(E))))
  return(S)
