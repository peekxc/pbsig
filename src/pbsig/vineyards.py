import numpy as np
import scipy.sparse as ss
from scipy.sparse import *
from typing import * 

from .persistence import * 

def permute_tr(A, i, type: str = "cols"):
  """ Transposes A via (i,i+1)"""
  if type == "cols":
    A[:,(i,i+1)] = A[:,(i+1,i)]
  elif type == "rows":
    A[(i,i+1),:] = A[(i+1,i),:]
  elif type == "both":
    permute_tr(A, i, "cols")
    permute_tr(A, i, "rows")

def cancel_pivot(A, i, j, piv: Optional[int] = None):
  """
  Attempts to use entries in A[:,i] to zero a lowest / "pivot" entry of A[:,j] in some field. 

  If piv is given (the row index of A[:,j] to set to zero), then A[piv,j] is zeroed out instead. 

  Otherwise, piv is inferred by the lowest-nonzero entry of A[:,j].

  If A[piv,i] = 0, then the pivot cannot be canceled, and the matrices are unmodified. 
  """
  ## Real-valued coefficients 
  if piv is None: 
    piv = low_entry(A, j)
  c, d = A[piv,i], A[piv,j]
  if (c == 0):
    return(0.0)
  s = -(d/c)
  A[:,j] += s*A[:,i]
  return(s)

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
        V2[:,l] += s*V2[:,k] if s != 0 else V2[:,k]
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
        R1[:,i+1] += s*R1[:,i] if s != 0 else R1[:,i]
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
        R1[:,i+1] += s*R1[:,i] if s != 0 else R1[:,i]
        # R1[:,i+1] = R1[:,i+1] + R1[:,i]
        # V1[:,i+1] = V1[:,i+1] + V1[:,i]
        permute_tr(R1, i, "cols")
        permute_tr(V1, i, "both")
        permute_tr(R2, i, "rows")
        s = cancel_pivot(R1, i, i+1)
        V1[:,i+1] += s*V1[:,i] if s != 0 else V1[:,i]
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
      R1[:,i+1] += s*R1[:,i] if s != 0 else R1[:,i]
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      s = cancel_pivot(R1, i, i+1)
      V1[:,i+1] += s*V1[:,i] if s != 0 else V1[:,i]
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


def linear_homotopy(f0: ArrayLike, f1: ArrayLike, plot_lines: bool = False, schedule: bool = False):
  assert len(f0) == len(f1)
  ind = np.argsort(f0)
  P, Q = np.c_[np.repeat(0.0, len(f0)),f0][ind,:], np.c_[np.repeat(1.0, len(f1)),f1][ind,:]
  from itertools import combinations 
  n = P.shape[0]
  results = []
  for i,j in combinations(range(n), 2):
    pq = line_intersection((P[i,:], Q[i,:]), (P[j,:], Q[j,:]))
    if not(pq is None):
      results.append((i,j,pq[0],pq[1]))
    # print((i,j))
  results = np.array(results)
  results = results[np.logical_and(results[:,2] >= 0.0, results[:,2] <= 1.0),:]
  results = results[np.argsort(results[:,2]),:]
  cross_points = results[:,[2,3]].astype(float)
  transpositions = results[:,[0,1]].astype(int)

  if plot_lines: 
    import matplotlib.pyplot as plt 
    qp = np.argsort(Q[:,1])
    plt.scatter(*P.T)
    plt.scatter(*Q.T)
    for i,j,k,l in zip(P[:,0],Q[:,0],P[:,1],Q[:,1]):
      plt.plot((i,j), (k,l), c='black')
    plt.scatter(*results[:,2:].T, zorder=30, s=5.5)
    for x,y,label in zip(P[:,0],P[:,1],range(n)): plt.text(x, y, s=str(label),ha='left')
    for x,y,label in zip(Q[qp,0],Q[qp,1],qp): plt.text(x, y, s=str(label),ha='right')

  if not(schedule):
    return(transpositions, cross_points)
  else:
    q = np.argsort(f1[np.argsort(f0)])
    S = schedule_transpositions(transpositions, q)
    return(S, cross_points[:,0])
  

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

def merge_sort(A):
  if len(A) <= 1:
      return A, 0    
  else:
      middle = (len(A)//2)
      left, count_left = merge_sort(A[:middle])
      right, count_right = merge_sort(A[middle:])
      result, count_result = merge(left,right)
      return result, (count_left + count_right + count_result)

def merge(a,b):
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
  if len(a) == 0:
    result = result + b
  else:
    result = result + a
  return result, count

# From: https://jamesmccaffrey.wordpress.com/2021/11/22/the-kendall-tau-distance-for-permutations-example-python-code/
def inversion_dist(a,b):
  assert all(np.unique(a) == np.fromiter(range(len(a)), int))
  # from scipy.stats import kendalltau
  # # w = kendalltau(np.array(a),np.array(b)).correlation*(len(a)*(len(a)-1))/2
  # # return(int(np.ceil(w/2)))
  # w = 1.0 - (kendalltau(np.array(a),np.array(b)).correlation + 1.0)/2
  # w *= (len(a)*(len(a)-1))/2
  # return(int(np.ceil(w/2)))
  # p1, p2 are 0-based lists or np.arrays permutations
  n = len(a)
  index_of = [None] * n  # lookup into p2
  for i in range(n):
    v = b[i]; index_of[v] = i

  d = 0  # raw distance = number pair mis-orderings
  for i in range(n):  # scan thru p1
    for j in range(i+1, n):
      if index_of[a[i]] > index_of[a[j]]: 
        d += 1
  normer = n * (n - 1) / 2.0  # total num pairs 
  nd = d / normer  # normalized distance
  return int(np.ceil(d))


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


def schedule(f0: ArrayLike, f1: ArrayLike):
  if not(all(np.argsort(f0) == np.argsort(f1))):
    F0 = dict(zip(range(len(f0)), f0))
    F1 = dict(zip(range(len(f1)), f1))
    a = np.fromiter(dict(sorted(F0.items(), key=lambda kv: (kv[1], kv[0]))).keys(), int)
    b = np.fromiter(dict(sorted(F1.items(), key=lambda kv: (kv[1], kv[0]))).keys(), int)

    ## New method: just use bubble sort and forget about forming the homotopy
    b_to_id = dict(zip(b, range(len(b)))) # mapp symbols in b to identity 
    p = np.array([b_to_id[s] for s in a])
    r_tr = schedule_transpositions(p, np.array(inversions(p))) # relative transpositions
    assert (r_tr.shape[0] == inversion_dist(a,b))
    return(r_tr)
  else:
    return(np.zeros((0, 2)))


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
  pp = p.copy()
  pp[[i,i+1]] = pp[[i+1,i]]
  return(pp)

# def schedule_inversion():
#   inversions(a,b)