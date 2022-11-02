# from math import comb
import numpy as np
import ot

from math import comb
from typing import Iterable
from itertools import combinations
from numpy.typing import ArrayLike
from . import rotate_S1
from .persistence import lower_star_ph_dionysus
from .utility import progressbar, shape_center, uniform_S1, PL_path, complex2points
from gudhi.wasserstein import wasserstein_distance

def pht_transformer(scale = ["directions", "diameter"], translate: str = "directions", nd: int = 32, **kwargs):
  assert nd % 2 == 0, "Number of directions must be even"
  # if center == "directions":
  V = np.array(list(uniform_S1(nd)))
  def _preprocess(X: ArrayLike) -> ArrayLike:
    u = shape_center(X, method="directions", V=V, **kwargs)
    L = -sum([min(X @ vi[:,np.newaxis]) for vi in V])
    X = X - u 
    X = (nd/L)*X
    return(X)
  return(_preprocess)

def pht_preprocess_pc(X: ArrayLike, nd: int = 32):
  P = pht_transformer(nd)
  return(P(X))

def pht_preprocess_path_2d(n_directions: int = 32, n_segments: int = 100):
  """
  Params: 
    n_directions := number of directions around S1 to use to center shape 
    n_segments := number of linear segments to discretize path by
  Return: 
    preprocess := function that takes as input a path and returns a set of 2d points centered at 0 / scaled barycentrically
  """
  V_dir = np.array(list(uniform_S1(n_directions)))
  def _preprocess(path):
    C = PL_path(path, k=n_segments) 
    P = complex2points([c.start for c in C])
    u = shape_center(P, method="directions", V=V_dir)
    P = P - u 
    L = sum([-np.min(P @ vi[:,np.newaxis]) for vi in V_dir])
    P = (1/L)*P
    return(P)
  return _preprocess

def pht_0(X: ArrayLike, E: ArrayLike, nd: int = 32, transform: bool = True, progress: bool = False, replace_inf: bool = False):
  if transform:
    X = pht_preprocess_pc(X, nd, transform=True)
  circle = rotate_S1(X, nd=nd, include_direction=False)
  circle_it = progressbar(circle, nd) if progress else circle 
  dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in circle_it]
  if replace_inf: 
    for i, fv in zip(range(len(dgms0)), rotate_S1(X, nd=nd, include_direction=False)):
      dgm = dgms0[i]
      which_inf = dgm[:,1] == np.inf
      dgm[which_inf,1] = max(fv)
      dgms0[i] = dgm
  return(dgms0)

def wasserstein_mod_rot(D0, D1, p: float = 1.0, **kwargs) -> float:
  """
  D0 := List of diagrams 
  D1 := List of diagrams
  """
  from pbsig.utility import rotate
  assert len(D0) == len(D1), "Lists containing diagrams should be equal"
  wd = lambda a,b: wasserstein_distance(a, b, matching=False, order=p, keep_essential_parts=True)
  wdists = []
  m = len(D0)
  for i in range(m): 
    D_rot = rotate(D0, i)
    wdists.append(sum([wd(d0,d1) for d0,d1 in zip(D_rot, D1)]))
  return(wdists[np.argmin(wdists)])

def pht_0_dist(X: Iterable, mod_rotation: bool = True, nd: int = 32, preprocess: bool = True, progress: bool = False, diagrams: bool = False):
  """
  X := iterable of (V, E) where V is point cloud of vertex positions and E are edges, or iterable of diagrams
  nd := number of directions
  """
  if not(diagrams): 
    shape_it = progressbar(X, count=len(X))
    D = list()
    for V,E in X:
      V = pht_preprocess_pc(V, nd, transform=True) if preprocess else V
      D.append(pht_0(V, E, nd, transform=False))
  else: # input is a set of diagrams
    D = X
  ns = len(D) # number of shapes 
  comb_it = progressbar(combinations(D, 2), comb(ns, 2)) if progress else combinations(D, 2)
  if mod_rotation:
    pht_dist = np.array([wasserstein_mod_rot(D0, D1) for D0, D1 in comb_it])
  else: 
    wd = lambda D0, D1: sum([wasserstein_distance(d0, d1) for d0,d1 in zip(D0, D1)])
    pht_dist = np.array([wd(D0, D1) for D0, D1 in comb_it])
  return(pht_dist)

from pbsig.betti import lower_star_betti_sig

def pbt_birth_death(X: Iterable[ArrayLike], E: Iterable[ArrayLike], nd: int = 32, preprocess: bool = True):
  if preprocess: 
    X = [pht_preprocess_pc(x, nd, transform=True) for x in X]
  F = np.array(list(rotate_S1(X, n=32, include_direction=False)))
  five_num = lambda x: (np.min(x), np.max(x), np.std(x), np.mean(x), np.median(x))
  info = np.vstack([five_num(F.min(axis=1)), five_num(F.max(axis=1)), five_num(F.flatten())])
  return(info)
    
def pbt_0_dist(X: Iterable[ArrayLike], E: Iterable[ArrayLike], nd: int = 32, preprocess: bool = True, progress: bool = False, **kwargs):
  assert len(X) == len(E), "Edges and X must have same length"
  if preprocess: 
    X = [pht_preprocess_pc(x, nd, transform=True) for x in X]
  ns = len(X)
  Sigs = []
  for x, e in progressbar(zip(X, E), ns):
    circle = rotate_S1(x, n=nd, include_direction=False)
    Sigs.append(lower_star_betti_sig(circle, e, x.shape[0], **kwargs))
  min_rot_euc = lambda s0, s1: min([sum(abs(np.roll(s0, i) - s1)) for i in range(len(s0))])
  comb_it = progressbar(combinations(Sigs, 2), comb(ns, 2)) if progress else combinations(Sigs, 2)
  pbt_dist = np.array([min_rot_euc(s0, s1) for s0, s1 in comb_it])
  return(pbt_dist)
