from .meta import *
# from .utility import is_repeateb
from scipy.interpolate import CubicSpline, UnivariateSpline
from more_itertools import pairwise, triplewise, chunked, peekable, spy
# from scipy.interpolate import splrep, insert
from splex import *
from splex.predicates import is_repeatable
from typing import * 
from scipy.sparse import diags

class ParameterizedLaplacian():
  def __init__(self, S: ComplexLike, p: int = 0):
    self.p_faces = np.array(list(faces(S, p)))
    self.q_faces = np.array(list(faces(S, p+1)))
    self.BM = boundary_matrix(S, p = p+1)
  
  def interpolate_family(self, F: Iterable[Callable], interval=(0.0, 1.0)) -> None: 
    """Interpolates a 1-parameter family of filtration values."""
    filter_values = [[f(self.p_faces), f(self.q_faces)] for f in F]
    n_parameters = len(filter_values)
    p_filter_values = np.array([f[0] for f in filter_values], dtype=np.float32)
    q_filter_values = np.array([f[1] for f in filter_values], dtype=np.float32)
    del filter_values

    ## Interpolate the filter path of each simplex (simplexwise) with a polynomial curve
    self.domain_ = interval
    domain_points = np.linspace(*interval, num=n_parameters) # knot vector
    self.p_splines_ = [CubicSpline(domain_points, p_fv) for p_fv in p_filter_values.T]
    self.q_splines_ = [CubicSpline(domain_points, q_fv) for q_fv in q_filter_values.T]
    
  def laplace(self, t: float, normed: bool = False):
    assert hasattr(self, "domain_") and hasattr(self, "q_splines_"), "Must fit to parameterized family first!"
    assert t >= self.domain_[0] and t <= self.domain_[1], f"Invalid time point 't'; must in the domain [{self.domain_[0]},{self.domain_[1]}]"
    from pbsig.linalg import pseudoinverse
    wp = np.array([pf(t) for pf in self.p_splines_])
    wq = np.array([qf(t) for qf in self.q_splines_])
    L = self.BM @ diags(wq) @ self.BM.T
    if normed: 
      deg = (diags(np.sign(wp)) @ L @ diags(np.sign(wp))).diagonal() ## retain correct nullspace
      L = (diags(np.sqrt(pseudoinverse(deg))) @ L @ diags(np.sqrt(pseudoinverse(deg)))).tocoo()
    else:
      L = (diags(np.sqrt(pseudoinverse(wp))) @ L @ diags(np.sqrt(pseudoinverse(wp)))).tocoo()
    return L
  
  def figure_curves(self):
    import bokeh


# def interpolate_family(S: ComplexLike, x: ArrayLike, y: Iterable[ArrayLike], bs: int = 1, method: str = "cubic", **kwargs):
#   """"""
#   y = list(y) if not(is_repeatable(y)) else y
#   assert is_repeatable(y), "y must be repeatable (a generator is not sufficient)"
#   head, y = spy(y)
#   assert len(head[0]) == len(S), "Family _y_ Must match the number of simplices"
#   # y = [codensity(alpha) for alpha in alpha_family]
#   # x = alpha_family
#   # assert len(x) == len(y), "Invalid x,y pair"
#   # yi = codensity(0.30)
#   # np.fromiter((yi(s) for s in S), dtype=np.float32)

#   ## Builds interpolated curves for each simplex 
#   BS = {}
#   for s_ind in chunked(range(len(S)), bs):
#     curves = np.vstack([yi[s_ind] for yi in y])
#     for i, c in zip(s_ind, curves):
#       BS[i] = CubicSpline(x, c, **kwargs)
#   return BS 


## TODO: expand functionality to allow time to be tuple or Iterable, then use time[0], time[-1] to 
## get bounds to allow for non-uniform discrtization of time. 
def interpolate_point_cloud(X: Iterable[ArrayLike], time: tuple = None, method: str = "cubic", **kwargs):
  """Interpolates a dynamic point cloud changing continuously with time. 
  
  Returns a Callable that continuously interpolates the point cloud over _time_. 
  """
  if time is not None:
    assert isinstance(time, tuple) and len(time) == 2
    assert isinstance(time[0], Number) and isinstance(time[1], Number)
  time_points = np.linspace(0,1,len(X)) if time is None else np.linspace(time[0],time[1],len(X))
  assert is_repeatable(X), "Iterable must be repeatable."

  ## Deduce the number of points in order to perform the interpolation point-wise
  head, _ = spy(X)
  n_pts = len(head[0])

  curves = {}
  for i in range(n_pts):
    varying_points = np.array([x[i] for x in X])
    curves[i] = CubicSpline(time, varying_points, **kwargs)

  return curves

class LinearFuncInterp:
  """Provides a sized-callable that merges set of interpolate functions into a single function f: [0,1] -> Any """
  def __init__(self, funcs: Iterable[Callable]) -> None:
    self.funcs = list(funcs)
    self.time_points = np.linspace(0,1,len(self.funcs)+1)
  
  def __len__(self) -> int:
    return len(self.funcs)

  def __call__(self, t: float) -> Any:
    if t >= 1.0: return self.funcs[-1](1.0)
    if t <= 0.0: return self.funcs[0](0.0)
    d = 1.0/len(self.funcs)
    end_ind = np.searchsorted(self.time_points, t)
    new_t = abs((t-self.time_points[end_ind-1])/d)
    return self.funcs[end_ind-1](new_t)

    # self.funcs[end_ind]
    # if ind < len(time_points):

    # else: 
    #   IP = np.array([c(time_points[-1]) for c in curves])
    #   return IP
