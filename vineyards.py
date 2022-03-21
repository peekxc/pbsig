import numpy as np
import scipy.sparse as ss
from scipy.sparse import *
from typing import * 
from persistence import * 

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
  Attempts to use entries in A[:,i] to zero a pivot entry of A[:,j] in some field. 

  If piv is given (the row index of A[:,j] to zero), then this is canceled instead. 

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
        V2[:,l] += s*V2[:,k]
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
      if low_R1[i] < low_R1[i+1]:
        s = cancel_pivot(R1, i, i+1)
        V1[:,i+1] += s*V1[:,i]
        #R1[:,i+1] = R1[:,i+1] + R1[:,i]
        #V1[:,i+1] = V1[:,i+1] + V1[:,i]
        permute_tr(R1, i, "cols")
        permute_tr(V1, i, "both")
        permute_tr(R2, i, "rows")
        return(4)
      else:
        # s = cancel_pivot(R1, i, i+1)
        s = cancel_pivot(V1, i, i+1, piv=i)
        R1[:,i+1] += s*R1[:,i]
        # R1[:,i+1] = R1[:,i+1] + R1[:,i]
        # V1[:,i+1] = V1[:,i+1] + V1[:,i]
        permute_tr(R1, i, "cols")
        permute_tr(V1, i, "both")
        permute_tr(R2, i, "rows")
        s = cancel_pivot(R1, i, i+1)
        V1[:,i+1] += s*V1[:,i]
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
      R1[:,i+1] += s*R1[:,i]
      permute_tr(R1, i, "cols")
      permute_tr(V1, i, "both")
      permute_tr(R2, i, "rows")
      s = cancel_pivot(R1, i, i+1)
      V1[:,i+1] += s*V1[:,i]
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

def linear_homotopy(f0: ArrayLike, f1: ArrayLike, plot_lines: bool = False):
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
  transpositions = results[:,:2].astype(int)

  pp = np.array(range(n), dtype=int)
  qp = np.argsort(Q[:,1])

  if plot_lines: 
    import matplotlib.pyplot as plt 
    plt.scatter(*P.T)
    plt.scatter(*Q.T)
    for i,j,k,l in zip(P[:,0],Q[:,0],P[:,1],Q[:,1]):
      plt.plot((i,j), (k,l), c='black')
    plt.scatter(*results[:,2:].T, zorder=30, s=5.5)
    for x,y,label in zip(P[:,0],P[:,1],pp): plt.text(x, y, s=str(label),ha='left')
    for x,y,label in zip(Q[qp,0],Q[qp,1],qp): plt.text(x, y, s=str(label),ha='right')

  ## Convert id transpositions to relative transpositions
  rtransp = np.zeros(shape=transpositions.shape, dtype=int)
  p_inv = np.argsort(pp)
  perm = pp.copy()
  for c,(i,j) in enumerate(transpositions):
    rtransp[c,:] = [p_inv[i],p_inv[j]]
    perm[[p_inv[j],p_inv[i]]] = perm[[p_inv[i],p_inv[j]]]
    # p_inv[[i,j]] = [j,i]
    p_inv = np.argsort(perm)
  
  assert np.all(qp == perm)
  assert np.all(np.abs(rtransp[:,0]-rtransp[:,1]) == 1)
  return(rtransp)