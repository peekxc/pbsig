import numpy as np
import scipy.sparse as ss
import copy
from scipy.sparse import *
from typing import * 
from array import array 
from itertools import combinations
from .combinatorial import longest_subsequence, comb_mod
from .persistence import * 
from .simplicial import *

_MOVE_STATS = { "n_right" : 0, "n_left" : 0, "n_cols_left" : 0, "n_cols_right" : 0 }
_VINE_STATS = { "n_col" : 0, "n_tr" : 0 }

def permutation_matrix(p: Sequence[int]):
  """ 
  Returns the column representation of a permutation 
  
  Given the word representation of a permutation 'p' the appropriate permutation matrix acts on a matrix 'A' via: 
  
  P @ A @ P.T

  where P.T applied on the right applies p to the columns, and P applied on the left applies p to the rows. 
  """
  n = len(p)
  P = coo_array((np.ones(n), (np.arange(n), p)), shape=(n,n), dtype=int)
  return P

def move_left_permutation(i: int, j: int, n: int) -> ArrayLike:
  """ """
  assert 0 <= i and i <= j and j <= n, "Must satisfy i <= j <= n"
  iind = np.arange(0, n)
  if i == j: return iind # identity
  bind = np.arange(i,j+1)
  iind[bind] = np.roll(bind, 1)
  return iind

def move_right_permutation(i: int, j: int, n: int) -> ArrayLike:
  """ 
  Returns the word representation (the arrangement) of the permutation that moves i to j, where i < j, in id(n).
  This has the effect of shifting all intermediate symbols between (i,j] down 1. 
  """
  assert 0 <= i and i <= j and j <= n, "Must satisfy i <= j <= n"
  iind = np.arange(0, n)
  if i == j: return iind # identity
  bind = np.arange(i,j+1)
  iind[bind] = np.roll(bind, -1)
  return iind


def permute_cylic(A, i, j, type: str = "cols", right: bool = True):
  """ Permutes A cyclically about [i,j] in-place """
  bind = np.arange(i,j+1)
  rind = np.roll(bind, -1 if right else 1)
  if hasattr(A, "ndim") and A.ndim == 2: 
    if type == "cols":
      A[:,bind] = A[:,rind]
    elif type == "rows":
      A[bind,:] = A[rind,:]
    elif type == "both":
      permute_cylic(A, i, j, "cols", right=right)
      permute_cylic(A, i, j, "rows", right=right)
  else:
    return np.array([A[x] for x in bind])

def permute_cylic_pure(A, i, j, type: str = "cols", right: bool = True):
  """ Permutes A cyclically about [i,j] in-place """
  rind = np.roll(np.arange(i,j+1), -1 if right else 1)
  if hasattr(A, "ndim") and A.ndim == 2: 
    if type == "cols":
      bind = np.arange(A.shape[1])
      bind[i:(j+1)] = rind
      return A[:,bind]
    elif type == "rows":
      bind = np.arange(A.shape[0])
      bind[i:(j+1)] = rind
      return A[bind,:]
    elif type == "both":
      A = A.copy()
      permute_cylic(A, i, j, "cols", right=right)
      permute_cylic(A, i, j, "rows", right=right)
      return A
  else:
    bind = np.arange(len(A))
    bind[i:(j+1)] = rind
    return np.array([A[x] for x in bind])

def permute_tr(A, i, type: str = "cols"):
  """ Transposes A via (i,i+1)"""
  if type == "cols":
    A[:,(i,i+1)] = A[:,(i+1,i)]
  elif type == "rows":
    A[(i,i+1),:] = A[(i+1,i),:]
  elif type == "both":
    permute_tr(A, i, "cols")
    permute_tr(A, i, "rows")

from scipy.sparse import coo_array
# cancel_column(DS, 1, DS[:,[0]])
def add_column(A, k, col):
  """ Adds column 'col' to A[:,k] """
  if np.issubdtype(A.dtype, np.floating):
    A[:,[k]] += col 
  elif np.issubdtype(A.dtype, np.integer): 
    if isinstance(A, np.ndarray):
      A[:,[k]] = (A[:,[k]] + col) % 2 ## Modulo2 
    else: 
      # assert isinstance(A, lil_array), "invalid A types"
      col = col.todense() if isinstance(col, spmatrix) else col 
      A[:,[k]] = (col + A[:,[k]].todense()) % 2 ## Modulo2 
    return 0
  else:
    raise ValueError(f"Unknown coefficient type {A.dtype}")

def cancel_column(A, k, col):
  """ Adds column 'col' to A[:,k], canceling the lowest one """
  if isinstance(A.dtype, numbers.Real) or np.issubdtype(A.dtype, float):
    s = float(low_entry(A[:,[k]]))/float(low_entry(col))
    A[:,[k]] -= s*col 
  elif isinstance(A.dtype, numbers.Integral) or np.issubdtype(A.dtype, np.integer): 
    ## Modulo2 
    if isinstance(A, np.ndarray):
      A[:,[k]] = (A[:,[k]] + col) % 2
    else: 
      assert isinstance(A, lil_array), "invalid A types"
      col = col.todense() if isinstance(col, spmatrix) else col 
      A[:,[k]] = (col + A[:,[k]].todense()) % 2
      # i, j = A[:,[k]].nonzero()
      # A[:,[k]] = coo_array((data % 2,(ind, [0]*len(ind))), shape=A[:,[k]].shape).tolil() # awful 
      #A[:,[k]] = lil_array(A.getcol(k).data % 2).T
    return 0
  else:
    raise ValueError(f"Unknown coefficient type {A.dtype}")

def cancel_pivot(A, i, j, piv: Optional[int] = None):
  """
  Attempts to use entries in A[:,i] to zero a lowest / "pivot" entry of A[:,j] in some field. 

  If piv is given (the row index of A[:,j] to set to zero), then A[piv,j] is zeroed out instead. 

  Otherwise, piv is inferred by the lowest-nonzero entry of A[:,j].

  If A[piv,i] = 0, then the pivot cannot be canceled, and the matrices are unmodified. 
  """
  piv = low_entry(A, j) if piv is None else piv
  c, d = A[piv,i], A[piv,j]
  if np.issubdtype(A.dtype, np.floating):
    ## Real-valued coefficients 
    if (c == 0): return(0.0)
    s = -(d/c)
    A[:,[j]] += s*A[:,[i]]
    return(s)
  elif np.issubdtype(A.dtype, np.integer): 
    ## Modulo2 
    if (c == 0): return(0)
    if (isinstance(A, np.ndarray)):
      A[:,[j]] = (A[:,[i]] + A[:,[j]]) % 2
    else:
      A[:,[j]] = (A[:,[i]] + A[:,[j]]).todense() % 2
    return 0
  else:
    raise ValueError(f"Unknown coefficient type {A.dtype}")

def transpose_rv(R: lil_array, V: lil_array, I: Iterable):
  """ 
  The vineyards algorithm 
  
  Critical cases: 2, 5, 7
  """
  assert R.shape[0] == V.shape[0], "Must be matching boundary matrices"
  assert R.shape[0] == R.shape[1] and V.shape[0] == V.shape[1], "Must be square"
  m = R.shape[1] # num simplices in filtration
  for i in I:
    _VINE_STATS['n_tr'] += 1
    piv = low_entry(R)
    pos = piv == -1
    if pos[i] and pos[i+1]:
      if V[i,i+1] != 0:
        # s = cancel_pivot(V, i, i+1, piv=i) 
        add_column(V, i+1, V[:,[i]])
        _VINE_STATS['n_col'] += 1
      if any(piv == i) and any(piv == i+1):
        k,l = np.flatnonzero(piv == i).item(), np.flatnonzero(piv == (i+1)).item()
        if R[i,l] != 0: # Case 1.1.1 and 1.1.2
          status = 1 if k < l else 2
          k,l = (k,l) if k < l else (l,k) # ensure k < l
          permute_tr(R, i, "both")
          permute_tr(V, i, "both")
          # s = cancel_pivot(R, k, l)
          # V[:,[l]] += s*V[:,[k]] if s != 0 else V[:,[k]]
          add_column(V, l, V[:,[k]])
          add_column(R, l, R[:,[k]])
          _VINE_STATS['n_col'] += 1
          yield status
        else: # Case 1.2 still! 
          permute_tr(R, i, "both")
          permute_tr(V, i, "both")
          yield 3 # should be same as below
      else: # Case 1.2
        permute_tr(R, i, "both")
        permute_tr(V, i, "both")
        yield 3
    elif not(pos[i]) and not(pos[i+1]):
      if V[i,i+1] != 0: 
        if piv[i] < piv[i+1]: # pivot in i is higher than pivot in i+1
          #s = cancel_pivot(V, i, i+1, piv=i) ## V1[:,i+1] |-> s*V1[:,i] + V1[:,i+1], where s := ()
          #R[:,[i+1]] += s*R[:,[i]] if s != 0 else R[:,[i]]
          add_column(V, i+1, V[:,[i]])
          add_column(R, i+1, R[:,[i]])
          permute_tr(R, i, "both")
          permute_tr(V, i, "both")
          _VINE_STATS['n_col'] += 1
          yield 4 
        else:
          # s = cancel_pivot(V, i, i+1, piv=i)
          # R[:,[i+1]] += s*R[:,[i]] if s != 0 else R[:,[i]]
          add_column(V, i+1, V[:,[i]])
          add_column(R, i+1, R[:,[i]])
          permute_tr(R, i, "both")
          permute_tr(V, i, "both")
          # s = cancel_pivot(R, i, i+1)
          # V[:,[i+1]] += s*V[:,[i]] if s != 0 else V[:,[i]]
          add_column(V, i+1, V[:,[i]])
          add_column(R, i+1, R[:,[i]])
          _VINE_STATS['n_col'] += 2
          yield 5
      else: # Case 2.2
        permute_tr(R, i, "both")
        permute_tr(V, i, "both")
        yield 6 
    elif not(pos[i]) and pos[i+1]:
      if V[i,i+1] != 0:
        ## Case 3.1
        # s = cancel_pivot(V, i, i+1, piv=i)
        # R[:,[i+1]] += s*R[:,[i]] if s != 0 else R[:,[i]]
        add_column(V, i+1, V[:,[i]])
        add_column(R, i+1, R[:,[i]])
        permute_tr(R, i, "both")
        permute_tr(V, i, "both")
        # s = cancel_pivot(R, i, i+1)
        # V[:,[i+1]] += s*V[:,[i]] if s != 0 else V[:,[i]]
        add_column(R, i+1, R[:,[i]])
        add_column(V, i+1, V[:,[i]])
        _VINE_STATS['n_col'] += 2
        yield 7
      else:
        permute_tr(R, i, "both")
        permute_tr(V, i, "both")
        yield 8
    elif pos[i] and not(pos[i+1]):
      if V[i,i+1] != 0:
        # s = cancel_pivot(V, i, i+1, piv=i)
        add_column(V, i+1, V[:,[i]])
        _VINE_STATS['n_col'] += 1
      permute_tr(R, i, "both")
      permute_tr(V, i, "both")
      yield 9


def transpose_dgm(R1, V1, R2, V2, i: int):
  assert (R1.shape[1] == V1.shape[0]), "Must be matching boundary matrices"
  assert i < (R1.shape[1]-1), "Invalid i"
  # handle case where R2,V2 empty
  m = R1.shape[1]
  pos = low_entry(R1) == -1
  if pos[i] and pos[i+1]:
    if V1[i,i+1] != 0:
      #V1[:,i+1] = V1[:,i+1] + V1[:,i]
      s = cancel_pivot(V1, i, i+1, piv=i) # todo: 
    low_R2 = low_entry(R2)
    if np.any(low_R2 == i) and np.any(low_R2 == (i+1)):
      k,l = np.flatnonzero(low_R2 == i).item(), np.flatnonzero(low_R2 == (i+1)).item()
      if R2[i,l] != 0:
        k,l = (k,l) if k < l else (l,k) # ensure k < l
        permute_tr(R1, i, "cols")
        permute_tr(R2, i, "rows")
        permute_tr(V1, i, "both")
        s = cancel_pivot(R2, k, l)
        V2[:,[l]] += s*V2[:,[k]] if s != 0 else V2[:,[k]]
        #V2[:,l] += s*V2[:,k]
        #R2[:,l] = R2[:,k] + R2[:,l]
        #V2[:,l] = V2[:,k] + V2[:,l]
        return(1)
      else:
        permute_tr(R1, i, "cols")
        permute_tr(R2, i, "rows")
        permute_tr(V1, i, "both")
        return(2)
    else:
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      return(3)
  elif not(pos[i]) and not(pos[i+1]):
    if V1[i,i+1] != 0: 
      low_R1 = low_entry(R1)
      if low_R1[i] < low_R1[i+1]: # pivot in i is higher than pivot in i+1
        # s = cancel_pivot(R1, i, i+1)
        s = cancel_pivot(V1, i, i+1, piv=i) ## V1[:,i+1] |-> s*V1[:,i] + V1[:,i+1], where s := ()
        #assert s != 0
        R1[:,[i+1]] += s*R1[:,[i]] if s != 0 else R1[:,[i]]
        #R1[:,i+1] = R1[:,i+1] + R1[:,i]
        #V1[:,i+1] = V1[:,i+1] + V1[:,i]
        permute_tr(R1, i, "cols")
        permute_tr(V1, i, "both")
        permute_tr(R2, i, "rows")
        return(4)
      else:
        # s = cancel_pivot(R1, i, i+1)
        s = cancel_pivot(V1, i, i+1, piv=i)
        # R1[:,i+1] += s*R1[:,i]
        R1[:,[i+1]] += s*R1[:,[i]] if s != 0 else R1[:,[i]]
        # R1[:,i+1] = R1[:,i+1] + R1[:,i]
        # V1[:,i+1] = V1[:,i+1] + V1[:,i]
        permute_tr(R1, i, "cols")
        permute_tr(V1, i, "both")
        permute_tr(R2, i, "rows")
        s = cancel_pivot(R1, i, i+1)
        V1[:,[i+1]] += s*V1[:,[i]] if s != 0 else V1[:,[i]]
        return(5)
    else: # Case 2.2
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      return(6)
  elif not(pos[i]) and pos[i+1]:
    if V1[i,i+1] != 0:
      ## Case 3.1
      s = cancel_pivot(V1, i, i+1, piv=i)
      R1[:,[i+1]] += s*R1[:,[i]] if s != 0 else R1[:,[i]]
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      s = cancel_pivot(R1, i, i+1)
      V1[:,[i+1]] += s*V1[:,[i]] if s != 0 else V1[:,[i]]
      return(7)
    else:
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      return(8)
  elif pos[i] and not(pos[i+1]):
    if V1[i,i+1] != 0:
      s = cancel_pivot(V1, i, i+1, piv=i)
      # V1[:,i+1] = V1[:,i+1] + V1[:,i]
    permute_tr(R1, i, "cols")
    permute_tr(V1, i, "both")
    permute_tr(R2, i, "rows")
    return(9)
  return(0)

def line_intersection(line1, line2):
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
  def det(a, b): return a[0] * b[1] - a[1] * b[0]
  div = det(xdiff, ydiff)
  if div == 0:
    return(None)
  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  return x, y


# def linear_homotopy(f0: ArrayLike, f1: ArrayLike, plot_lines: bool = False, schedule: bool = False):
#   assert len(f0) == len(f1)
#   ind = np.argsort(f0)
#   P, Q = np.c_[np.repeat(0.0, len(f0)),f0][ind,:], np.c_[np.repeat(1.0, len(f1)),f1][ind,:]
#   from itertools import combinations 
#   n = P.shape[0]
#   results = []
#   for i,j in combinations(range(n), 2):
#     pq = line_intersection((P[i,:], Q[i,:]), (P[j,:], Q[j,:]))
#     if not(pq is None):
#       results.append((i,j,pq[0],pq[1]))
#     # print((i,j))
#   results = np.array(results)
#   results = results[np.logical_and(results[:,2] >= 0.0, results[:,2] <= 1.0),:]
#   results = results[np.argsort(results[:,2]),:]
#   cross_points = results[:,[2,3]].astype(float)
#   transpositions = results[:,[0,1]].astype(int)

#   if plot_lines: 
#     import matplotlib.pyplot as plt 
#     qp = np.argsort(Q[:,1])
#     plt.scatter(*P.T)
#     plt.scatter(*Q.T)
#     for i,j,k,l in zip(P[:,0],Q[:,0],P[:,1],Q[:,1]):
#       plt.plot((i,j), (k,l), c='black')
#     plt.scatter(*results[:,2:].T, zorder=30, s=5.5)
#     for x,y,label in zip(P[:,0],P[:,1],range(n)): plt.text(x, y, s=str(label),ha='left')
#     for x,y,label in zip(Q[qp,0],Q[qp,1],qp): plt.text(x, y, s=str(label),ha='right')

#   if not(schedule):
#     return(transpositions, cross_points)
#   else:
#     q = np.argsort(f1[np.argsort(f0)])
#     S = schedule_transpositions(transpositions, q)
#     return(S, cross_points[:,0])
  

## Convert id transpositions to relative position transpositions
def schedule_transpositions(p: ArrayLike, transpositions: ArrayLike):
  #p = np.fromiter(range(n), dtype=int)
  if len(transpositions) == 0:
    return(transpositions)
  rtransp = np.zeros(shape=transpositions.shape, dtype=int)
  n = len(p)
  p_inv = np.argsort(p)
  perm = p.copy()
  for c,(i,j) in enumerate(transpositions):
    rtransp[c,:] = [p_inv[i],p_inv[j]]
    perm[[p_inv[j],p_inv[i]]] = perm[[p_inv[i],p_inv[j]]]
    # p_inv[[i,j]] = [j,i]
    # p_inv[i], p_inv[j] = p_inv[j], p_inv[i]
    p_inv = np.argsort(perm)
  
  assert all(np.abs(rtransp[:,0]-rtransp[:,1]) == 1)
  assert all(p_inv == np.fromiter(range(n), dtype=int))
  return(rtransp)

def _merge_sort(A: MutableSequence) -> Tuple[MutableSequence, int]:
  if len(A) <= 1:
    return A, 0    
  else:
    middle = (len(A) // 2)
    left, count_left = _merge_sort(A[:middle])
    right, count_right = _merge_sort(A[middle:])
    result, count_result = _merge(left,right)
    return result, (count_left + count_right + count_result)

def _merge(a: MutableSequence, b: MutableSequence) -> Tuple[MutableSequence, int]:
  result = []
  count = 0
  while len(a) > 0 and len(b) > 0:
    if a[0] <= b[0]:   
      result.append(a[0])
      a.remove(a[0])
    else:
      result.append(b[0])
      b.remove(b[0])
      count += len(a)
  result = result + b if len(a) == 0 else result + a
  return result, count

def inversion_dist(a: MutableSequence, b: Optional[MutableSequence] = None, method: str = ["merge_sort", "pairwise"]):
  ## First convert to 0-index integer arrays
  a = list(np.argsort(np.argsort(a)))
  b = list(np.argsort(np.argsort(b))) if b is not None else None

  if b is not None and method == "merge_sort" or method == ["merge_sort", "pairwise"]: 
    assert all(np.unique(a) == np.unique(b))  
    b_map = { sb : i for i, sb in enumerate(b) }
    a = [b_map[sa] for sa in a]
    sa, d = _merge_sort(a)
  elif b is not None and method == "pairwise": 
    assert all(np.unique(a) == np.unique(b))
    IB = np.argsort(b)
    d = sum([IB[a[i]] > IB[a[j]] for i,j in combinations(range(len(a)), 2)])
  elif b is None: 
    sa, d = _merge_sort(a)
  else: 
    raise ValueError("unknown input")
  return(d)

def is_discordant(pairs: Iterable, p: Sequence, q: Sequence):
  p_inv, q_inv = np.argsort(p), np.argsort(q)
  c1 = lambda a,b: (p_inv[a] == q_inv[a]) and (p_inv[b] == q_inv[b])
  c2 = lambda a,b: np.sign(p_inv[a] - p_inv[b]) == -np.sign(q_inv[a] - q_inv[b])
  return([c2(a,b) and not(c1(a,b)) for a,b in pairs])

def pair_is_discordant(a: int, b: int, p: Sequence, q: Sequence) -> bool:
  ap, bp = p.index(a), p.index(b)
  aq, bq = q.index(a), q.index(b)
  return(False if ap == aq and bp == bq else np.sign(ap - bp) == -np.sign(aq - bq))

def plot_linear_homotopy(F0: Dict, F1: Dict):
  assert F0.keys() == F1.keys(), "Invalid dictionaries"
  import matplotlib.pyplot as plt
  n = len(F0)
  f0 = np.fromiter(F0.values(), dtype=float)
  f1 = np.fromiter(F1.values(), dtype=float)
  fig = plt.figure(figsize=(8,4), dpi=320)
  ax = fig.gca()
  ax.set_xlim(-0.1, 1.1)
  ax.set_ylim(min([min(f0), min(f1)]), max([max(f0), max(f1)]))
  ax.scatter(np.repeat(0, n), f0, s=3.5)
  ax.scatter(np.repeat(1, n), f1, s=3.5)
  # all((ax.text(-0.025, y, s=c, ha='center', va='center') for c, y in F0.items()))
  # all((ax.text(1.025, y, s=c, ha='center', va='center') for c, y in F1.items()))
  for i in F0.keys():
    pp, qq = (0, F0[i]), (1, F1[i])
    ax.plot(*np.vstack((pp, qq)).T, c='blue', linewidth=0.5)

def linear_homotopy(f: Union[ArrayLike, Dict], g: Union[ArrayLike, Dict], interval: tuple = (0, 1), plot: bool = False):
  """
  Decomposes a linear homotopy h: R x [0,1] -> R between two sequences 'f' and 'g' into an ordered set of adjacent transpositions. 
  
  That is, given two real-valued sequences 'f' and 'g' of equal size, this function first construct 'h' such that:

  h(i,t) = (1-t)*f[i] + t*g[i], for t in 'interval'

  then, by varying all values of 't' in the given interval, adjacent transposition are found as 'crossings' in the homotopy.  
  These crossings are ordered and reported, such that 'f' is _sorted_ into 'g'.

  Parameters: 
    f := function values, given as numpy array or dict[Any, float], starting at interval[0]
    g := function values, given as numpy array or dict[Any, float], ending at interval [1]
    interval := the interval to simulate the homotopy across. Defaults to [0,1].
    plot := whether to plot the the homotopy. Useful for debugging/visualizing small cases. 

  Returns: 
    (tr, x) := array of integers (a, b, ...) giving the adjacent transposition (a, a+1), (b, b+1), ... which sorts f |-> g, and their doman values 'x' 

  If f,g are arrays, then the homotopy occurs between (f[i], g[i]) for all i. Otherwise the key of f are matched with those of g.  
  """
  pairwise = lambda C: zip(C, C[1:])

  ## Convert to dictionaries f: < simplex > -> < filtration value > 
  if isinstance(f, np.ndarray) and isinstance(g, np.ndarray):
    assert len(f) == len(g), "Invalid inputs; should be equal size"
    f = { c : f for c, f in enumerate(f) }
    g = { c : f for c, f in enumerate(g) }
  elif isinstance(f, Dict) and isinstance(g, Dict):
    assert f.keys() == g.keys(), "f and g should have the same key-sets"
  elif isinstance(f, MutableFiltration) and isinstance(g, MutableFiltration):
    f = { v:k for k,v in f.items() }
    g = { v:k for k,v in g.items() }
    assert f.keys() == g.keys(), "f and g should have the same key-sets"
  else: 
    raise ValueError("Unknown input for f,g detected")
  
  ## Sort the map to line up the filtration values
  F0 = dict(sorted(f.items(), key=lambda kv: kv[1])) if not(is_sorted(f.values())) else f.copy()
  F1 = dict(sorted(g.items(), key=lambda kv: kv[1])) if not(is_sorted(g.values())) else g.copy()
  
  ## Reindex the simplices to permutations
  g_map = { s : i for i,s in enumerate(g.keys()) }
  p = list(np.array([g_map[s] for s in f.keys()], dtype=np.int32))
  q = list(np.fromiter(g_map.values(), dtype=np.int32))
  F0 = dict(zip(p, F0.values()))
  F1 = dict(zip(q, F1.values()))
  # p, q = list(F0.keys()), list(F1.keys())

  ## Create permutations representing the symbols of each filtration value
  n = len(f)
  f0 = np.fromiter(F0.values(), dtype=float)
  f1 = np.fromiter(F1.values(), dtype=float)
  assert is_sorted(f0) and is_sorted(f1)
  
  ## Usually for debugging
  if plot: plot_linear_homotopy(F0, F1)

  ## Compute the adjacent transpositions in a valid order following a linear homotopy
  p_inv, q_inv = np.argsort(p), np.argsort(q)
  discordant = lambda i,j: np.sign(p_inv[i] - p_inv[j]) != np.sign(q_inv[i] - q_inv[j])
  s, e = interval
  tr, dom_vals = array('I'), array('f')
  eps = 10*np.finfo(float).resolution
  while p != q:
    cross_f = np.zeros(n-1) ## how early does a pair cross
    # tau_dist = inversion_dist(p, q)
    for i, (sa, sb) in enumerate(pairwise(p)):
      # pair_dc = pair_is_discordant(sa,sb,p,q)
      #if pair_is_discordant(sa,sb,p,q): # and inversion_dist(swap(p, i),q) < tau_dist: # discordant pair
      if discordant(sa,sb):
        ## Try to handle degenerate cases: they are discordant, so they need to swap!
        if np.isclose(F0[sa], F0[sb]):
          cross_f[i] = 0.0
        elif np.isclose(F1[sa], F1[sb]):
          cross_f[i] = 1.0
        else:
          int_pt = intersection(line((s, F0[sa]), (e, F1[sa])), line((s, F0[sb]), (e, F1[sb])))
          assert int_pt[0] >= (s-eps) and int_pt[0] <= (e+eps) # check if intersection lies beyond [s,e]
          if int_pt[0] <= (s-eps) or int_pt[0] >= (e+eps):
            # print(f"L1: {F0[sa]} <--> {F1[sa]}, L2: {F0[sb]} <--> {F1[sb]} (X: {int_pt[0]},{int_pt[1]}")
            cross_f[i] = np.inf 
          else:
            cross_f[i] = int_pt[0]
      else: 
        cross_f[i] = np.inf
    assert not(all(cross_f == np.inf)), "Failed to find a valid crossing "
    ci = np.argmin(cross_f)
    if ci != np.inf:
      tr.append(ci)
      dom_vals.append(cross_f[ci])
      # tr_s.append((p[ci], p[ci+1]))
    else: 
      raise ValueError("invalid transposition case encountered")
    # p = swap(p, ci)
    ## Update inverse maps
    p_inv[p[ci]] += 1
    p_inv[p[ci+1]] -= 1
    p[ci], p[ci+1] = p[ci+1], p[ci]
  assert p == q, "Failed to sort permutations"
  return(np.asarray(tr, dtype=int), np.asarray(dom_vals, dtype=float))

def inversions(elements):
  A = np.fromiter(iter(elements), dtype=np.int32)
  inv = []
  swapped = False
  for n in range(len(A)-1, 0, -1):
    for i in range(n):
      if A[i] > A[i+1]:
        swapped = True
        inv.append([A[i], A[i+1]])
        A[i], A[i + 1] = A[i + 1], A[i]       
    if not swapped:
      break
  return(inv)

def line(p1, p2):
  A, B, C = (p1[1] - p2[1]), (p2[0] - p1[0]), (p1[0]*p2[1] - p2[0]*p1[1]) 
  return A, B, -C

def intersection(L1, L2):
  D  = L1[0] * L2[1] - L1[1] * L2[0]
  Dx = L1[2] * L2[1] - L1[1] * L2[2]
  Dy = L1[0] * L2[2] - L1[2] * L2[0]
  if D != 0:
    x = Dx / D
    y = Dy / D
    return x,y
  else:
    return False

## Swaps elements at positions [i,i+1]. Returns a new array
def swap(p, i):
  assert i < (len(p)-1)
  pp = copy.deepcopy(p)
  pp[i], pp[i+1] = pp[i+1], pp[i]
  return(pp)

# def schedule_inversion():
#   inversions(a,b)

# From: https://jamesmccaffrey.wordpress.com/2021/11/22/the-kendall-tau-distance-for-permutations-example-python-code/
# def inversion_dist(a,b):
#   assert all(np.unique(a) == np.fromiter(range(len(a)), int))
#   # from scipy.stats import kendalltau
#   # # w = kendalltau(np.array(a),np.array(b)).correlation*(len(a)*(len(a)-1))/2
#   # # return(int(np.ceil(w/2)))
#   # w = 1.0 - (kendalltau(np.array(a),np.array(b)).correlation + 1.0)/2
#   # w *= (len(a)*(len(a)-1))/2
#   # return(int(np.ceil(w/2)))
#   # p1, p2 are 0-based lists or np.arrays permutations
#   n = len(a)
#   index_of = [None] * n  # lookup into p2
#   for i in range(n):
#     v = b[i]; index_of[v] = i

#   d = 0  # raw distance = number pair mis-orderings
#   for i in range(n):  # scan thru p1
#     for j in range(i+1, n):
#       if index_of[a[i]] > index_of[a[j]]: 
#         d += 1
#   normer = n * (n - 1) / 2.0  # total num pairs 
#   nd = d / normer  # normalized distance
#   return int(np.ceil(d))
# normer = n * (n - 1) / 2.0  # total num pairs 
# nd = d / normer  # normalized distance
# return int(np.ceil(d))
# def inversions(a,b):
#   assert all(np.unique(a) == np.fromiter(range(len(a)), int))
#   n = len(a)
#   index_of = [None] * n  # lookup into p2
#   for i in range(n):
#     v = b[i]; index_of[v] = i

#   inv = []
#   for i in range(n):  # scan thru p1
#     for j in range(i+1, n):
#       if index_of[a[i]] > index_of[a[j]]: 
#         inv.append([a[i], a[j]])

#   return inv
# def schedule(f0: ArrayLike, f1: ArrayLike):
#   if not(all(np.argsort(f0) == np.argsort(f1))):
#     F0 = dict(zip(range(len(f0)), f0))
#     F1 = dict(zip(range(len(f1)), f1))
#     a = np.fromiter(dict(sorted(F0.items(), key=lambda kv: (kv[1], kv[0]))).keys(), int)
#     b = np.fromiter(dict(sorted(F1.items(), key=lambda kv: (kv[1], kv[0]))).keys(), int)

#     ## New method: just use bubble sort and forget about forming the homotopy
#     b_to_id = dict(zip(b, range(len(b)))) # mapp symbols in b to identity 
#     p = np.array([b_to_id[s] for s in a])
#     r_tr = schedule_transpositions(p, np.array(inversions(p))) # relative transpositions
#     assert (r_tr.shape[0] == inversion_dist(a,b))
#     return(r_tr)
#   else:
#     return(np.zeros((0, 2)))


def restore_right(R: lil_array, V: lil_array, I: Sequence[int]) -> tuple:
  if len(I) == 0: return None 
  dL, dR, dV = low_entry(R, I[0]), R[:,[I[0]]], V[:,[I[0]]]
  for k in I[1:]:
    tL, tR, tV = low_entry(R, k), R[:,[k]], V[:,[k]] # temporary 
    add_column(R, k, dR)
    add_column(V, k, dV)
    if tL < dL: 
      dL, dR, dV = tL, tR, tV
  return dR, dV  

def move_right(R: lil_array, V: lil_array, i: int, j: int, copy: bool = False) -> None:
  assert i < j, f"Invalid pair ({i},{j}) given (i >= j)"
  R, V = (R.copy(), V.copy()) if copy else (R, V)
  piv = low_entry(R) 
  IR = V[i,i:(j+1)].nonzero()[1] if isinstance(V, spmatrix) else np.flatnonzero(V[i,i:(j+1)] != 0) # relative indices
  I = np.arange(i,j+1)[IR] # global indices
  J = np.flatnonzero(np.logical_and(piv >= i, piv <= j)) # R[[i],:].todense() != 0
  J = np.array([l for l in J if R[i,l] != 0])
  #J_check = np.flatnonzero(low_entry(permute_cylic_pure(R, i, j, "both")) == j)
  #assert all(J_check == J), "J index check failed"
  dR, dV = restore_right(R, V, I)
  restore_right(R, V, J) # this should not affect the number of non-reduced columns
  _MOVE_STATS["n_cols_right"] += len(I) + len(J)
  permute_cylic(R, i, j, "both") ## change if full boundary matrix is used
  permute_cylic(V, i, j, "both")
  R[:,[j]], V[:,[j]] = permute_cylic_pure(dR, i, j, "rows"), permute_cylic_pure(dV, i, j, "rows")
  return (R, V) if copy else None
  
def move_left(R, V, j, i, copy: bool = False):
  """ Moves column j of (R,V) to position i, shifting intermediate columns up one. """
  assert i < j, f"Invalid pair ({i},{j}) given (i >= j)"
  R,V = (R.copy(), V.copy()) if copy else (R,V)

  ## Clear column V[:,j] in rows (i:j+1) to allow permutation
  I = []
  for k in reversed(range(i,j)):
    if V[k,j] != 0:
      add_column(V, j, V[:,k]) # V[:,j] += col_V(k) 
      add_column(R, j, R[:,k]) # R[:,j] += col_R(k)
      I.append(k)
  _MOVE_STATS["n_cols_left"] += len(I) 

  ## Permute both R and V
  ## NOTE: Do *not* re-assign to R and V, otherwise this becomes a pure function 
  permute_cylic(R, i, j, type="both", right=False)
  permute_cylic(V, i, j, type="both", right=False)

  ## Restore left 
  ## Can still guarentee non-reduced low entries will be in [i,j]
  ## Can we predict the 
  ind_R = np.arange(R.shape[0])
  low_R = low_entry(R)
  can_R = [(l,r) for l,r in combinations(ind_R, 2) if low_R[l] == low_R[r] and low_R[l] != -1]
  while len(can_R) > 0:
    (l,r) = max(can_R, key=lambda lr: low_R[lr[0]])
    # print((l,r))
    add_column(R, r, R[:,[l]])
    add_column(V, r, V[:,[l]])
    low_R = low_entry(R)
    can_R = [(l,r) for l,r in combinations(ind_R, 2) if low_R[l] == low_R[r] and low_R[l] != -1]
    _MOVE_STATS["n_cols_left"] += 1
  return (R,V) if copy else None


## LIS/LCS -> move scheduling
def move_schedule(p: Sequence[int], method: str = "nearest", verbose: bool = False) -> Sequence[int]:
  assert isinstance(method, str) and method in ["nearest", "random", "greedy"], f"Invalid schedule heuristic {method}"
  assert all(np.sort(p) == np.arange(len(p))), "Invalid permutation; should be arrangement (word permutation) of [0,n-1]"
  
  ## Prep the LIS, its complement set, and p's extended sequence
  m = len(p)
  lis = np.array(longest_subsequence(p))
  com = np.setdiff1d(p, lis)
  L = np.array([-1] + list(p) + [m])

  ## Define successor, predecessor, and position functions
  succ = lambda L, i: L[np.flatnonzero(i < L)[0]] if any(i < L) else m
  pred = lambda L, i: L[np.flatnonzero(L < i)[-1]] if any(L < i) else -1
  pos = lambda L, i: np.flatnonzero(L == i)[0]

  ## Used to generate candidate moves 
  def query_move(symbol, lst, lis):
    i = pos(lst, symbol)
    j = pos(lst, pred(lis, symbol)) ## b
    k = pos(lst, succ(lis, symbol)) ## e
    t = j if i < j else k    ## target index, using nearest heuristic 
    return (i,t)

  ## Generate the schedule
  schedule = []
  while not(is_sorted(L)):

    ## Apply the heuristic to find the next move. 
    ## Each move should increase the LIS monotonically
    if method == "nearest":
      d = com[0]
      i,t = query_move(d, L, lis)
    elif method == "random":
      d = np.random.choice(com, size=1).item()
      i,t = query_move(d, L, lis)
    elif method == "greedy": 
      Q = [query_move(s, L, lis) for s in com]
      I = np.array(Q).flatten().astype(np.int32)
      i,t = Q[np.argmin(comb_mod.interval_cost(I))]
    else: 
      assert isinstance(method, Callable)
      Q = [query_move(s, L, lis) for s in com]
      i,t = method(Q)
    if verbose: print(f"L: {L}, LIS: {lis}, complement: {com}, symbol: {L[i]}")
      
    ## Move a symbol in the complement set and update the LIS and L 
    sym = L[i]
    assert sym in com, "Invalid symbol chosen"
    com = np.setdiff1d(com, sym)
    L = permute_cylic_pure(L, min(i,t), max(i,t), right=i < t)
    lis = np.sort(np.append(lis, sym))
    schedule.append((i,t))

  ## Validate that the schedule does indeed sort to the identity permutation 
  S = p.copy()
  for i,j in schedule:
    i,j = i-1,j-1
    S = permute_cylic_pure(S, min(i,j), max(i,j), right=i < j)
  assert is_sorted(S)
  schedule = np.array(schedule, dtype=np.int32) - 1 # flatten for c++ later
  return schedule

def move_stats(reset: bool = False) -> None:
  if reset:
    _MOVE_STATS["n_right"] = 0
    _MOVE_STATS["n_left"] = 0
    _MOVE_STATS["n_cols_left"] = 0
    _MOVE_STATS["n_cols_right"] = 0
  return _MOVE_STATS.copy()

def vineyards_stats(reset: bool = False) -> None:
  if reset:
    _VINE_STATS["n_cols"] = 0
    _VINE_STATS["n_tr"] = 0
  return _VINE_STATS.copy()

from scipy.sparse import spmatrix
def update_lower_star(K: MutableFiltration, R: spmatrix, V: spmatrix, f: Callable, vines: bool = False, **kwargs):
  """ 
  Updates the (R,V) factors of the persistence decomposition of given filtration 'K' to reflect the filter 'f' 

  Also updates K via reindexing! 
  
  """
  assert isinstance(V.dtype, numbers.Integral) or np.issubdtype(V.dtype, np.integer), "Only works mod2"
  assert isinstance(R.dtype, numbers.Integral) or np.issubdtype(R.dtype, np.integer), "Only works mod2"
  assert isinstance(K, MutableFiltration), "Invald filtration given"
  L = MutableFiltration(K.values(), f = f)
  if vines:
    schedule, _ = linear_homotopy(K, L)
    status = [s for s in transpose_rv(R, V, schedule)]
    K.reindex(f)
  else:
    L_map = { s : i for i,s in enumerate(L.values()) }
    K_perm = np.array([L_map[s] for s in K.values()], dtype=np.int32)
    schedule = move_schedule(K_perm, **kwargs)
    for i,j in schedule:
      if i < j:
        move_right(R, V, i, j)
        _MOVE_STATS["n_right"] += 1
      else:
        move_left(R, V, i, j) 
        _MOVE_STATS["n_left"] += 1
    K.reindex(f)
    
  