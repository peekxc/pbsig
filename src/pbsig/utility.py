import sys
import numpy as np 

from itertools import *
from operator import le
from typing import * 
from numpy.typing import ArrayLike
from math import comb

def pairwise(S: Iterable): 
  a, b = tee(S)
  next(b, None)
  return zip(a, b)

def window(S: Iterable, n: int = 2):
  "Returns a sliding window (of width n) over data from the iterable"
  "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
  it = iter(S)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for elem in it:
    result = result[1:] + (elem,)
    yield result

def rotate(S: Iterable, n: int = 1):
  from collections import deque
  items = deque(S)
  items.rotate(n)
  return iter(items)

def cycle_window(S: Iterable, offset: int = 1, w: int = 2):
  return window(islice(cycle(S), len(S)+offset), w)

def partition_envelope(f: Callable, threshold: float, interval: Tuple = (0, 1), lower: bool = False):
  """
  Partitions the domain of a real-valued function 'f' into intervals by evaluating the pointwise maximum of 'f' and the constant function g(x) = threshold. 
  The name 'envelope' is because this can be seen as intersecting the upper-envelope of 'f' with 'g'

  Parameters: 
    f := Callable that supports f.operator(), f.derivative(), and f.roots() (such as the Splines from scipy)
    threshold := cutoff-threshold
    interval := interval to evalulate f over 
    lower := whether to evaluate the lower envelope instead of the upper 

  Return: 
    intervals := (m x 3) nd.array giving the intervals that partition the corresponding envelope
  
  Each row (b,e,s) of intervals has the form: 
    b := beginning of interval
    e := ending of interval
    s := 1 if in the envelope, 0 otherwise 

  """
  assert isinstance(interval, Tuple) and len(interval) == 2

  ## Partition a curve into intervals at some threshold
  in_interval = lambda x: x >= interval[0] and x <= interval[1]
  crossings = np.fromiter(filter(in_interval, f.solve(threshold)), float)

  ## Determine the partitioning of the upper-envelope (1 indicates above threshold)
  intervals = []
  if len(crossings) == 0:
    is_above = f(0.50).item() >= threshold
    intervals.append((0.0, 1.0, 1 if is_above else 0))
  else:
    if crossings[-1] != 1.0: 
      crossings = np.append(crossings, 1.0)
    b = 0.0
    df = f.derivative(1)
    df2 = f.derivative(2)
    for c in crossings:
      grad_sign = np.sign(df(c))
      if grad_sign == -1:
        intervals.append((b, c, 1))
      elif grad_sign == 1:
        intervals.append((b, c, 0))
      else: 
        accel_sign = np.sign(df2(c).item())
        if accel_sign > 0: # concave 
          intervals.append((b, c, 1))
        elif accel_sign < 0: 
          intervals.append((b, c, 0))
        else: 
          raise ValueError("Unable to detect")
      b = c
  
  ### Finish up and return 
  intervals = np.array(intervals)
  if lower: 
    intervals[:,2] = 1 - intervals[:,2]
  return(intervals)
  
## From: https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, count=None, prefix="", size=60, out=sys.stdout): # Python3.6+
  count = len(it) if count == None else count 
  def show(j):
    x = int(size*j/count)
    print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
  show(0)
  for i, item in enumerate(it):
    yield item
    show(i+1)
  print("\n", flush=True, file=out)

def is_sorted(L: Iterable, compare: Callable = le):
  a, b = tee(L)
  next(b, None)
  return all(map(compare, a, b))

# def is_sorted(A: Iterable, key=lambda x: x):
#   for i, el in enumerate(A[1:]):
#     if key(el) < key(A[i]): # i is the index of the previous element
#       return False
#   return True

def smoothstep(lb: float = 0.0, ub: float = 1.0, order: int = 1, down: bool = False):
  """
  Maps [lb, ub] -> [0, 1] via a smoother version of a (up) step function. When down=False and lb=ub, the step functions look like: 

  down = False       |    down = True
  1:      o------    |    1: -----* 
  0: -----*          |    0:      o------  

  When |lb - ub| > 0, the returned function is a differentiable relaxation the above step function(s). 

  Parameters: 
    lb := lower bound in the domain to begin the step 
    ub := upper bound in the domain to end the step 
    order := smoothness parameter 
    down := whether to make the step a down step. Default to False (makes an up-step).
  
  Returns a vectorized function S(x) satisfying one of: 
    1. S(x) = 0 for all x <= lb, 0 < S(x) < 1 for all lb < x < ub, and S(x) = 1 for all x >= ub, if down = False
    2. S(x) = 1 for all x <= lb, 0 < S(x) < 1 for all lb < x < ub, and S(x) = 0 for all x >= ub, if down = True
  """
  assert ub >= lb, "Invalid input"
  if lb == ub:
    if down: 
      def _ss(x: float): return 1.0 if x <= lb else 0.0
    else:
      def _ss(x: float): return 0.0 if x <= lb else 1.0
  else: 
    d = (ub-lb)
    assert d > 0, "Must be positive distance"
    def _ss(x: float):
      if (x <= lb): return(1.0 if down else 0.0)
      if (x >= ub): return(0.0 if down else 1.0)
      y = (x-lb)/d 
      return (1.0 - (3*y**2 - 2*y**3)) if down else 3*y**2 - 2*y**3
  return(np.vectorize(_ss))

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

def unrank_comb(r: int, k: int, n: int):
  result = np.zeros(k, dtype=int)
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return(result)

def unrank_combs(R: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([unrank_C2(r, n) for r in R], dtype=int))
  else: 
    return(np.array([unrank_comb(r, k, n) for r in R], dtype=int))

def rank_comb(c: Tuple, k: int, n: int):
  c = np.array(c, dtype=int)
  #index = np.sum(comb((n-1)-c, np.flip(range(1, k+1))))
  index = np.sum([comb(cc, kk) for cc,kk in zip((n-1)-c, np.flip(range(1, k+1)))])
  return(int((comb(n, k)-1) - int(index)))

def rank_combs(C: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([rank_C2(c[0], c[1], n) for c in C], dtype=int))
  else: 
    return(np.array([rank_comb(c, k, n) for c in C], dtype=int))

def edges_from_triangles(triangles: ArrayLike, nv: int):
  ER = np.array([[rank_C2(*t[[0,1]], n=nv), rank_C2(*t[[0,2]], n=nv), rank_C2(*t[[1,2]], n=nv)] for t in triangles])
  ER = np.unique(ER.flatten())
  E = np.array([unrank_C2(r, n=nv) for r in ER])
  return(E)

def expand_triangles(nv: int, E: ArrayLike):
  ''' k=2 Expansion: requires Gudhi '''
  from gudhi import SimplexTree
  st = SimplexTree()
  for v in range(nv): st.insert([int(v)], 0.0)
  for e in E: st.insert(e)
  st.expansion(2)
  triangles = np.array([s[0] for s in st.get_simplices() if len(s[0]) == 3])
  return(triangles)

def scale_diameter(X: ArrayLike, diam: float = 1.0):
  from scipy.spatial.distance import pdist
  vec_mag = np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1))
  vec_mag[vec_mag == 0] = 1 
  VM = np.reshape(np.repeat(vec_mag, X.shape[1]), (len(vec_mag), X.shape[1]))
  Xs = (X / VM) * diam*(VM/(np.max(vec_mag)))
  return(Xs)


def shape_center(X: ArrayLike, method: str = ["pointwise", "bbox", "hull", "polygon", "directions"], V: Optional[ArrayLike] = None, atol: float = 1e-12):
  """
  Given a set of (n x d) points 'X' in d dimensions, returns the (1 x d) 'center' of the shape, suitably defined. 

  Each type of center varies in cost and definition, but each can effectively be interpreted as some kind of 'barycenter'. In particular: 

  barycenter := point-wise barycenter of 'X', equivalent to arithmetic mean of the points 
  box := barycenter of bounding box of 'X' 
  hull := barycenter of convex hull of 'X'
  polygon := barycenter of 'X', when 'X' traces out the boundary of a closed polygon in R^2*
  directions := barycenter of 'X' projected onto a set of rotationally-symmetric direction vectors 'V' in S^1
  
  The last options 'directions' uses an iterative process (akin to meanshift) to find the barycenter 'u' satisfying: 

  np.array([min((X-u) @ v)*v for v in V]).sum(axis=0) ~= [0.0, 0.0]
  """
  import warnings
  n, d = X.shape[0], X.shape[1]
  method = "directions" if V is not None else method 
  if method == "barycenter" or method == ["barycenter", "directions", "bbox", "hull"]:
    return(X.mean(axis=0))
  elif method == "hull":
    from scipy.spatial import ConvexHull
    return(X[ConvexHull(X).vertices,:].mean(axis=0))
  elif method == "directions":
    assert V is not None and isinstance(V, np.ndarray) and V.shape[1] == d
    ## Check V is rotationally symmetric
    rot_sym = np.allclose([min(np.linalg.norm(V + v, axis=1)) for v in V], 0.0, atol=atol)
    warnings.warn("Warning: supplied vectors 'V' not rotationally symmetric up to supplied tolerance")
    original_center = X.mean(axis=0)
    cost: float = np.inf
    while not(np.isclose(cost, 0.0, atol=atol)):
      Lambda = [np.min(X @ vi[:,np.newaxis]) for vi in V]
      U = np.vstack([s*vi for s, vi in zip(Lambda, V)])
      center_diff = U.mean(axis=0)
      X = X - center_diff
      cost = min([
        np.linalg.norm(np.array([min(X @ vi[:,np.newaxis])*vi for vi in V]).sum(axis=0)),
        np.linalg.norm(center_diff)
      ])
    return(original_center - X.mean(axis=0))
  elif method == "bbox":
    min_x, max_x = X.min(axis=0), X.max(axis=0)
    return((min_x + (max_x-min_x)/2))
  elif method == "polygon":
    from shapely.geometry import Polygon
    P = Polygon(X)
    return np.array(list(P.centroid.coords)).flatten()
  else: 
    raise ValueError("Invalid method supplied")

def winding_distance(X: ArrayLike, Y: ArrayLike):
  """ 
  Returns the minimum 'winding' distance between sequences of points (X,Y) defining closed curves in the plane homeomorphic to S^1
  
  Given two arrays (X, Y) of size (n,m) = (|X|,|Y|) representing 'circle-ish outlines', i.e. representing shapes whose edge sets EX satisfy:
  
  EX = [(X[0,:], X[1,:]), ..., (X[-1,], X[0])] (and similar with Y)
  
  This function: 
  1. finds a set of N=max(n,m) equi-spaced points along the boundaries (EX,EY) 
  2. cyclically rotates the N points from (1) along EX to best-align the N points along EY (i.e. w/ minimum squared euclidean distance)
  3. computes the integrated distances between the PL-curves traced by (EX, EY) 

  Since the curves are PL, (3) reduces to a sequence of irregular quadrilateral area calculations. 
  """ 
  from scipy.spatial import ConvexHull
  from shapely.geometry import Polygon
  from itertools import cycle
  from pyflubber.closed import prepare
  A, B = prepare(X, Y)
  N = A.shape[0] # should match B 
  edge_pairs = cycle(zip(pairwise(A), pairwise(B)))
  int_diff = 0.0
  for (p1,p2), (q1,q2) in islice(edge_pairs, N+1):
    ind = ConvexHull([p1,p2,q1,q2]).vertices
    int_diff += Polygon(np.vstack([p1,p2,q1,q2])[ind,:]).area
  return(int_diff)

def shift_curve(line, reference, objective: Optional[Callable] = None):
  min_, best_offset = np.inf, 0
  if objective is None: 
    objective = lambda X,Y: np.linalg.norm(X-Y)
  for offset in range(len(line)):
    deviation = objective(reference, np.roll(line, offset, axis=0))
    if deviation < min_:
      min_ = deviation
      best_offset = offset
  if best_offset:
    line = np.roll(line, best_offset, axis=0)
  return line

def procrustes_cc(A: ArrayLike, B: ArrayLike, do_reflect: bool = True, matched: bool = False, preprocess: bool = False):
  """ Procrustes distance between closed curves """
  from procrustes.rotational import rotational
  from pbsig.pht import pht_preprocess_pc
  from pyflubber.closed import prepare
  if preprocess:
    A, B = pht_preprocess_pc(A), pht_preprocess_pc(B)
  if do_reflect:
    R_refl = np.array([[-1, 0], [0, 1]])
    A1,B1 = procrustes_cc(A, B, do_reflect=False, matched=matched, preprocess=False)
    A2,B2 = procrustes_cc(A @ R_refl, B, do_reflect=False, matched=matched, preprocess=False)
    return (A1, B1) if np.linalg.norm(A1-B1, 'fro') < np.linalg.norm(A2-B2, 'fro') else (A2, B2)
  else:
    if matched:
      R = rotational(A, B, pad=False, translate=False, scale=False)
      # return np.linalg.norm((A @ R.t) - B, 'fro')
      return A @ R.t, B
    else: 
      ## Fix clockwise / counter-clockwise orientation of points
      A_p, B_p = prepare(A, B)
      
      ## Shift points along shape until minimum procrustes distance is obtained
      pro_error = lambda X,Y: rotational(X, Y, pad=False, translate=False, scale=False).error
      os = np.argmin([pro_error(A_p, np.roll(B_p, offset, axis=0)) for offset in range(A_p.shape[0])])
      A, B = A_p, np.roll(B_p, os, axis=0)
      return procrustes_cc(A, B, matched=True)

def procrustes_dist_cc(A: ArrayLike, B: ArrayLike, **kwargs):
  A_p, B_p = procrustes_cc(A, B, **kwargs)
  return np.linalg.norm(A_p - B_p, 'fro')

def PL_path(path, k: int): 
  from svgpathtools import parse_path, Line, Path, wsvg
  arc_lengths = np.array([np.linalg.norm(seg.length()) for seg in path])
  if any(arc_lengths == 0):
    path = list(filter(lambda p: p.length() > 0.0, path))
    for j in range(len(path)-1):
      if path[j].end != path[j+1].start:
        path[j].end = path[j+1].start
    path = Path(*path)
  arc_lengths = np.array([np.linalg.norm(seg.length())for seg in path])
  assert(all(arc_lengths > 0)), "Invalid shape detected: lines with 0-length found and not handled"
  A = np.cumsum(arc_lengths)
  p_cc = np.linspace(0, max(A), k)
  idx = np.digitize(p_cc, A+np.sqrt(np.finfo(float).resolution))
  L_points = []
  for i, pp in zip(idx, p_cc):
    t = pp/A[i] if i == 0 else (pp-A[i-1])/(A[i]-A[i-1])
    L_points.append(path[i].point(t))
  connect_the_dots = [Line(p0, p1) for p0, p1 in pairwise(L_points)]
  if any(connect_the_dots[-1].end != connect_the_dots[0].start): #not(path.iscontinuous()) or path.isclosed():
    #connect_the_dots.append(Line(L_points[-1], L_points[0]))
    connect_the_dots.append(Line(connect_the_dots[-1].end, connect_the_dots[0].start))
  new_path = Path(*connect_the_dots)
  return(new_path)

def simplify_outline(S: ArrayLike, k: int):
  from svgpathtools import parse_path, Line, Path, wsvg
  Lines = [Line(p, q) for p,q in pairwise(S)]
  Lines.append(Line(Lines[-1].end, Lines[0].start))
  return(PL_path(Path(*Lines), k))

def offset_curve(path, offset_distance, steps=1000):
  """
  Takes in a Path object, `path`, and a distance, `offset_distance`, and outputs an piecewise-linear approximation of the 'parallel' offset curve.
  """
  from svgpathtools import parse_path, Line, Path, wsvg
  nls = []
  for seg in path:
    ct = 1
    for k in range(steps):
      t = k / steps
      offset_vector = offset_distance * seg.normal(t)
      nl = Line(seg.point(t), seg.point(t) + offset_vector)
      nls.append(nl)
  connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
  if path.isclosed():
    connect_the_dots.append(Line(nls[-1].end, nls[0].end))
  offset_path = Path(*connect_the_dots)
  return offset_path

import heapq
import numpy as np

# From: https://stats.stackexchange.com/questions/459601/are-these-any-existing-implementation-of-l1-isotonic-regression-in-python
def isotonic_regress(y: ArrayLike, w: Optional[ArrayLike] = None):
  """Finds a non-decreasing fit for the specified `y` under L1 norm.

  The O(n log n) algorithm is described in:
  "Isotonic Regression by Dynamic Programming", Gunter Rote, SOSA@SODA 2019.

  Args:
    y: The values to be fitted, 1d-numpy array.
    w: The loss weights vector, 1d-numpy array.

  Returns:
    An isotonic fit for the specified `y` which minimizies the weighted
    L1 norm of the fit's residual.
  """
  if w is None: 
    w = np.ones(len(y))
  h = []  # max heap of values
  p = np.zeros_like(y)  # breaking position
  for i in range(y.size):
    a_i = y[i]
    w_i = w[i]
    heapq.heappush(h, (-a_i, 2 * w_i))
    s = -w_i
    b_position, b_value = h[0]
    while s + b_value <= 0:
      s += b_value
      heapq.heappop(h)
      b_position, b_value = h[0]
    b_value += s
    h[0] = (b_position, b_value)
    p[i] = -b_position
  z = np.flip(np.minimum.accumulate(np.flip(p)))  # right_to_left_cumulative_min
  return z

def complex2points(x): 
  return(np.c_[np.real(x), np.imag(x)])

def uniform_S1(n: int = 10):
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)+(np.pi/2)
  for x,y in zip(np.cos(theta), np.sin(theta)):
    yield np.array([x,y])


def spectral_bound(V, E, type: str = ["graph", "laplacian"]):
  """
  Returns bounds on the spectral radius of either the Graph G = (V, E) or the Graph Laplacian L = D(G) - A(G)
  Based on the simple bounds from: "Bounds on the (Laplacian) spectral radius of graphs"
  """
  import networkx as nx
  G = nx.Graph()
  G.add_nodes_from(V)
  G.add_edges_from(E)
  bound = 0
  if (isinstance(type, list) and type == ["graph", "laplacian"]) or type == "graph":
    bound = np.sqrt(max([sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()]))
  elif type == "laplacian":
    bound = np.sqrt(2)*np.sqrt(max([G.degree(v)**2 + sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()])) 
  else: 
    raise ValueError("invalid graph type")
  return(bound)

