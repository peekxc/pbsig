from .meta import *
# from .utility import is_repeateb
from scipy.interpolate import CubicSpline, UnivariateSpline
from more_itertools import pairwise, triplewise, chunked, peekable, spy
# from scipy.interpolate import splrep, insert
from splex.predicates import is_repeatable

def interpolate_family(S: ComplexLike, x: ArrayLike, y: Iterable[ArrayLike], bs: int = 1, method: str = "cubic", **kwargs):
  """"""
  y = list(y) if not(is_repeatable(y)) else y
  assert is_repeatable(y), "y must be repeatable (a generator is not sufficient)"
  head, y = spy(y)
  assert len(head[0]) == len(S), "Family _y_ Must match the number of simplices"
  # y = [codensity(alpha) for alpha in alpha_family]
  # x = alpha_family
  # assert len(x) == len(y), "Invalid x,y pair"
  # yi = codensity(0.30)
  # np.fromiter((yi(s) for s in S), dtype=np.float32)

  ## Builds interpolated curves for each simplex 
  BS = {}
  for s_ind in chunked(range(len(S)), bs):
    curves = np.vstack([yi[s_ind] for yi in y])
    for i, c in zip(s_ind, curves):
      BS[i] = CubicSpline(x, c, **kwargs)
  return BS 
