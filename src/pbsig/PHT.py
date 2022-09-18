# from math import comb
from numpy.typing import ArrayLike
from pbsig import rotate_S1
from pbsig.persistence import lower_star_ph_dionysus
from pbsig.utility import progressbar

def pht_0(X: ArrayLike, E: ArrayLike, nd: int = 32, transform: bool = True, progress: bool = False):
  circle = rotate_S1(X, n=nd, include_direction=False)
  circle_it = progressbar(circle, nd) if progress else circle 
  dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in circle_it]
  return(dgms0)