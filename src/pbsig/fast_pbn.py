from typing import * 

import numpy as np 
from numpy.typing import ArrayLike

import _pbn as pbn

def testing_1():
  return pbn.ph0_lower_star

# pbc_ls := Persistent Betti Curve over a Lower Star filtration 
def pbc_ls(fv: ArrayLike, E: ArrayLike, collapse: bool = True, lex_sort: bool = True, max_death: Any = ["inf", "max"]) -> ArrayLike:
  import pbsig
  from pbsig.pht import pht_preprocess_pc, rotate_S1
  from pbsig.datasets import mpeg7
  from pbsig.utility import cycle_window
  dataset = mpeg7()
  X = dataset[('turtle',2)]
  X = pht_preprocess_pc(X, nd=32)
  a,b,c,d = 0.20, 0.40, 0.60, 0.80
  E = np.array(list(cycle_window(range(X.shape[0]))))
  rotate_S1(X, nd = 132, include_direction=False)
  f = next(  rotate_S1(X, nd = 132, include_direction=False))
  f[E].max(axis=1)

