## Typing support 
from typing import *
from numpy.typing import ArrayLike 

## Module imports
import numpy as np
import math
import numbers 
import scipy.sparse as sps

## Function/structure imports
from scipy.sparse import *
from array import array
from itertools import *
from scipy.special import *
from scipy.spatial.distance import *

## Relative package imports
import _boundary as boundary
from .apparent_pairs import *
from .utility import *
from .simplicial import *

_perf = {
  "n_col_adds" : 0,
  "n_field_ops": 0
}
def H1_boundary_matrix(vertices: ArrayLike, edges: Iterable, coboundary: bool = False, dtype = int, sign_pattern: tuple = (1, -1)):
  p = np.argsort(vertices)        ## inverse permutation 
  v_sort = np.array(vertices)[p]  ## use p for log(n) lookup; this is [n] if all V unique
  V, E = len(vertices), len(edges)
  r_sign_pattern = tuple(reversed(sign_pattern))
  data_val, row_ind, col_ind = np.zeros(2*E, dtype=dtype), np.zeros(2*E, dtype=int), np.zeros(2*E, dtype=int)
  for i, (u,v) in enumerate(edges):
    u_ind, v_ind = np.searchsorted(v_sort, [u, v])
    riu, riv = int(p[u_ind]), int(p[v_ind]) # row index 
    # can use p[u], p[v] if V unique and == [n]
    data_val[2*i], data_val[2*i+1] = sign_pattern if riu < riv else r_sign_pattern
    row_ind[2*i], row_ind[2*i+1] = riu, riv
    col_ind[2*i], col_ind[2*i+1] = int(i), int(i)
  if coboundary:
    D = coo_array((data_val, (col_ind, row_ind)), shape=(E, V), dtype=dtype).tolil(copy=False)
  else: 
    D = coo_array((data_val, (row_ind, col_ind)), shape=(V, E), dtype=dtype).tolil(copy=False)
  
  ## TODO: make coboundary computation more efficient 
  # if coboundary:
  #   D = csc_matrix(np.flipud(np.fliplr(D.A)).T)
  return(D)

def H2_boundary_matrix(edges: Iterable, triangles: Iterable, coboundary: bool = False, N = "max", dtype = int, sign_pattern: tuple = (1, -1, 1)):
  if N == "max": N = np.max(np.array(edges).flatten())

  E_ranks = np.array([rank_C2(u, v, N) for (u,v) in edges], dtype=int)
  p = np.argsort(E_ranks)        ## inverse permutation 
  E_sort = E_ranks[p]            ## use p for log(n) lookup 
  E, T = len(E_ranks), len(triangles) # todo: fix 
  
  data_val, row_ind, col_ind = np.zeros(3*T, dtype=dtype), np.zeros(3*T, dtype=int), np.zeros(3*T, dtype=int)
  for i, (u,v,w) in enumerate(triangles):
    uv_ind, uw_ind, vw_ind = np.searchsorted(E_sort, [rank_C2(u, v, N), rank_C2(u, w, N), rank_C2(v, w, N)])
    data_val[3*i], data_val[3*i+1], data_val[3*i+2] = sign_pattern
    row_ind[3*i], row_ind[3*i+1], row_ind[3*i+2] = int(p[uv_ind]), int(p[uw_ind]), int(p[vw_ind])
    col_ind[3*i], col_ind[3*i+1], col_ind[3*i+2] = int(i), int(i), int(i)
  if coboundary:
    D = coo_array((data_val, (col_ind, row_ind)), shape=(T, E), dtype=dtype).tolil(copy=False)
  else: 
    D = coo_array((data_val, (row_ind, col_ind)), shape=(E, T), dtype=dtype).tolil(copy=False)  
  # D = csc_matrix((data_val, (row_ind, col_ind)), shape=(E, T), dtype=dtype)
  
  # ## TODO: make coboundary computation more efficient 
  # if coboundary:
  #   D = csc_matrix(np.flipud(np.fliplr(D.A)).T)
  return(D)

# from scipy.sparse import coo_array
# def boundary_matrix(K: Union[List, ArrayLike], p: Optional[Union[int, tuple]] = None):
#   """
#   Returns the ordered p-th boundary matrix of 'K'
#   """
#   from collections.abc import Sized
#   if isinstance(K, MutableFiltration):
#     assert p is None
#     # K.validate() ## needed to ensure faces exist
#     I,J,X = [],[],[] # row, col, data 
#     simplices = list(K.values(expand=True))
#     for (j,s) in enumerate(simplices):
#       if s.dimension() > 0:
#         I.extend([simplices.index(f) for f in s.faces(s.dimension()-1)])
#         J.extend(repeat(j, s.dimension()+1))
#         X.extend(islice(cycle([1,-1]), s.dimension()+1))
#     D = coo_array((X, (I,J)), shape=(len(simplices), len(simplices))).tolil(copy=False)
#     return D
#   elif isinstance(K, SimplicialComplex):
#     if p is None:
#       I,J,X = [],[],[] # row, col, data 
#       simplices = list(K) ## TODO: check if K has .index() method and use that, if provided
#       for (j,s) in enumerate(simplices):
#         if s.dimension() > 0:
#           I.extend([simplices.index(f) for f in s.faces(s.dimension()-1)])
#           J.extend(repeat(j, s.dimension()+1))
#           X.extend(islice(cycle([1,-1]), s.dimension()+1))
#       D = coo_array((X, (I,J)), shape=(len(simplices), len(simplices))).tolil(copy=False)
#       return D
#     elif isinstance(p, Integral):
#       K_lex = sorted(iter(K), key=lambda s: (len(s), tuple(s), s))
#       p_simplices = list(filter(lambda s: s.dimension() == p, K_lex))
#       p_faces = list(filter(lambda s: s.dimension() == p - 1, K_lex))
#       I,J,X = [],[],[] # row, col, data 
#       for (j,s) in enumerate(p_simplices):
#         if s.dimension() > 0:
#           I.extend([p_faces.index(f) for f in s.faces(s.dimension()-1)])
#           J.extend(repeat(j, s.dimension()+1))
#           X.extend(islice(cycle([1,-1]), s.dimension()+1))
#       D = coo_array((X, (I,J)), shape=(len(p_faces), len(p_simplices))).tolil(copy=False)
#       return D
#     elif isinstance(p, tuple):
#       return (boundary_matrix(K, pi) for pi in p)
#     else: 
#       raise ValueError("Invalid input")
  # elif isinstance(p, Iterable):
  #   if f is None: 
  #     f = [None]*len(p)
  #   return(boundary_matrix(K, p_, f_) for p_, f_ in zip(p, f))
  # else:
  #   print("why is this portion called? ")
    # if p == 0:
    #   nv = len(K['vertices'])
    #   D = csc_matrix((0, nv), dtype=np.float64).tolil()
    #   return D
    # elif p == 1: 
    #   D = H1_boundary_matrix(K['vertices'], K['edges'], coboundary=False)
    #   return D
    # elif p == 2:
    #   D = H2_boundary_matrix(K['edges'], K['triangles'], N = len(K['vertices']), coboundary=False)
    #   return D
    # elif p == 3:
    #   nt = len(K['triangles'])
    #   assert not('quads' in K)
    #   D = csc_matrix((nt, 0), dtype=np.float64).tolil()
    #   return D
    # elif p is None:
    #   # Return full boundary matrix
    #   nv, ne, nt = len(K['vertices']), K['edges'].shape[0], K['triangles'].shape[0]
    #   m = nv + ne + nt
    #   D1 = H1_boundary_matrix(K['vertices'], K['edges'])
    #   D2 = H2_boundary_matrix(K['edges'], K['triangles'])
    #   from scipy.sparse import bmat, dok_array
    #   B0, B1, B2 = dok_array((nv,nv)), dok_array((ne,ne)), dok_array((nt,nt))
    #   D = bmat([[B0, D1, dok_array((nv,nt))], [dok_array(D1.shape).T, B1, D2], [dok_array((nt, nv)), dok_array((nt,ne)), B2]])
    #   return D 
    # else: 
    #   raise ValueError(f"Unknown value p={p} supplied")

# def rips_boundary(X: ArrayLike, p: int, threshold: float):
#   pw_dist = pdist(X)
#   n = X.shape[0]
#   edge_ind = np.flatnonzero(pw_dist <= threshold)
#   edges = unrank_combs(edge_ind, n=n, k=2)
#   # edge_weights = D[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]
#   # D1.data = D1.data*np.maximum(b - np.repeat(edge_weights,2), 0)
#   if sorted: 
#     edge_weights = pw_dist[edge_ind]
#     ind = np.argsort(edge_weights)
#     D1 = D1[:,ind]
#     edge_weights = edge_weights[ind]

# def rips_H1(X: ArrayLike, K: List, b: float, sorted=False):
#   """ K := Rips complex """
#   E, T = len(K['edges']), len(K['triangles'])
#   D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'])
#   edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]
#   D1.data = D1.data*np.maximum(b - np.repeat(edge_weights,2), 0)
#   if sorted: 
#     ind = np.argsort(edge_weights)
#     D1 = D1[:,ind]
#     edge_weights = edge_weights[ind]
#   return(D1, edge_weights)

def weighted_H1(X: ArrayLike, K: List, sorted=False):
  """ K := Rips complex """
  E, T = len(K['edges']), len(K['triangles'])
  D1 = H1_boundary_matrix(vertices=K['vertices'], edges=K['edges'])
  edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in K['edges']], dtype=int)]
  if sorted: 
    ind = np.argsort(edge_weights)
    D1, edge_weights = D1[:,ind], edge_weights[ind] 
  return(D1, edge_weights)

def weighted_H2(X: ArrayLike, K: List, sorted=False):
  n, D = X.shape[0], pdist(X)
  tri_weight = lambda T: np.max([D[rank_C2(T[0],T[1],n)], D[rank_C2(T[0],T[2],n)], D[rank_C2(T[1],T[2],n)]])
  tri_weights = np.array([tri_weight(tri) for tri in K['triangles']])
  D2 = H2_boundary_matrix(edges=K['edges'], triangles=K['triangles'], N = X.shape[0])
  if sorted: 
    ind = np.argsort(tri_weights)
    D2, tri_weights = D2[:,ind], tri_weights[ind]
  return(D2, tri_weights)

def inverse_choose(x: int, k: int):
	assert k >= 1, "k must be >= 1" 
	if k == 1: return(x)
	if k == 2:
		rng = np.array(list(range(int(np.floor(np.sqrt(2*x))), int(np.ceil(np.sqrt(2*x)+2) + 1))))
		final_n = rng[np.nonzero(np.array([math.comb(n, 2) for n in rng]) == x)[0].item()]
	else:
		# From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
		if x < 10**7:
			lb = (math.factorial(k)*x)**(1/k)
			potential_n = np.array(list(range(int(np.floor(lb)), int(np.ceil(lb+k)+1))))
			idx = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[idx]
		else:
			lb = np.floor((4**k)/(2*k + 1))
			C, n = math.factorial(k)*x, 1
			while n**k < C: n = n*2
			m = (np.nonzero( np.array(list(range(1, n+1)))**k >= C )[0])[0].item()
			potential_n = np.array(list(range(int(np.max([m, 2*k])), int(m+k+1))))
			ind = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[ind]
	return(final_n)

def is_distance_matrix(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0. '''
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

def is_pairwise_distances(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a 1-d array of pairwise distances '''
	x = np.array(x, copy=False) # don't use asanyarray here
	if x.ndim > 1: return(False)
	n = inverse_choose(len(x), 2)
	return(x.ndim == 1 and n == int(n))

def is_point_cloud(x: ArrayLike) -> bool: 
	''' Checks whether 'x' is a 2-d array of points '''
	return(isinstance(x, np.ndarray) and x.ndim == 2)

def as_dist_matrix(x: ArrayLike, metric="euclidean") -> ArrayLike:
	if is_pairwise_distances(x):
		n = inverse_choose(len(x), 2)
		assert n == int(n)
		D = np.zeros(shape=(n, n))
		D[np.triu_indices(n, k=1)] = x
		D = D.T + D
		return(D)
	elif is_distance_matrix(x):
		return(x)
	else:
		raise ValueError("x should be a distance matrix or a set of pairwise distances")

def is_dist_like(x: ArrayLike):
	return(is_distance_matrix(x) or is_pairwise_distances(x))

def is_index_like(x: ArrayLike):
	return(np.all([isinstance(el, numbers.Integral) for el in x]))

def enclosing_radius(a: ArrayLike) -> float:
	''' Returns the smallest 'r' such that the Rips complex on the union of balls of radius 'r' is contractible to a point. '''
	assert is_distance_matrix(a)
	return(np.min(np.amax(a, axis = 0)))

def rips(X: ArrayLike, diam: float = np.inf, p: int = 1):
  ''' 
  Returns dictionary containing simplices and possibly weights of Rips d-complex of point cloud 'X' up to dimension 'd'.

  Parameters: 
    X := point cloud data 
    diam:= diameter parameter for the Rips complex
    p := dimension of boundary matrix to create
  '''
  assert is_point_cloud(X) or is_pairwise_distances(X), "Invalid format for rips"
  D = pdist(X) if is_point_cloud(X) else X
  N = inverse_choose(len(D), 2)
  rank_tri = lambda T : np.array([rank_C2(T[0],T[1],N), rank_C2(T[0],T[2],N), rank_C2(T[1],T[2],N)])
  Vertices, Edges, Triangles = np.arange(N, dtype=int), np.array([], dtype=int), array('I')
  if p == 0: 
    pass
  elif p == 1:
    Edges = np.flatnonzero(D <= diam) # done
  elif p == 2: 
    Edges = np.flatnonzero(D <= diam)
    for T in combinations(range(N), 3):
      e_ranks = rank_tri(T)
      ind = np.searchsorted(Edges, e_ranks)
      if np.all(ind < len(Edges)) and np.all(Edges[ind] == e_ranks):
        Triangles.extend(array('I', T))
  else: 
    raise ValueError("invalid p")
  result = {
    'vertices' : Vertices, 
    'edges' : np.array([unrank_C2(x,N) for x in Edges], dtype=int).reshape((len(Edges), 2)),
    'triangles' : np.asarray(Triangles).reshape((int(len(Triangles)/3), 3))
  }
  return(result)

def persistent_betti(D1, D2, i: int, j: int, summands: bool = False):
  """
  Computes the p-th persistent Betti number Bij of a pair of boundary matrices. 

  Bij = dim(Zp(Ki)) - dim(Zp(Ki) \cap Bp(Kj))

  Parameters:
    D1 := p-th filtered boundary matrix 
    D2 := (p+1) filtered boundary matrix 
    i := Birth index (integer)
    j := Death index (integer)
    summands := if True, returns a tuple (a, b, c) where the persistent Betti number B = a - b - c. Defaults to False. See details. 

  Given filtered boundary matrices (D1, D2) of dimensions (n,m) and (m,l), respectively, and 
  indices i \in [1, m] and j \in [1, l], returns the persistent Betti number 
  representing the number of p-dimensional persistent homology groups born at 
  or before index i (in D1) that were not destroyed by inclusion of the first j 
  (p+1)-simplices (D2). Note that (i,j) are given here as 1-based, index-coordinates. 
  Typically, the indices i and j are derived from geometric scaling parameters. 

  If summands = True, then a tuple (a, b, c) is returned where: 
    a := dim(Cp(Ki))
    b := dim(B{p-1}(Ki))
    c := dim(Zp(Ki) \cap Bp(Kj))
  The corresponding persistent Betti number is Bij = a - (b + c). 
  This can be useful for assessing the size of the group in the denominator (c).
  """
  #assert i < j, "i must be less than j"
  i, j = int(i), int(j)
  if (i == 0): 
    return((0, 0, 0, 0) if summands else 0)
  t1 = D1[:,:i].shape[1] # i, D1.shape[1]
  t2 = 0 if np.prod(D1[:,:i].shape) == 0 else np.linalg.matrix_rank(D1[:,:i].A)
  D2_tmp = D2[:,:j]
  t3_1 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
  D2_tmp = D2_tmp[i:,:] # note i is 1-based, so this is like (i+1)
  t3_2 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
  return((t1, t2, t3_1, t3_2) if summands else t1 - t2 - t3_1 + t3_2)

def persistent_betti_rips(X: ArrayLike, b: float, d: float, p: int = 1, **kwargs):
  """ 
  X := point cloud or set of pairwise distances
  b := birth diameter threshold
  d := death diameter threshold 
  p := unused 
  kwargs 
  """
  assert is_point_cloud(X) or is_pairwise_distances(X), "Invalid format for rips"
  assert b <= d, "Birth time must be less than death time"
  D = pdist(X) if is_point_cloud(X) else X
  D1, (vw, ew) = rips_boundary(D, p=1, diam=d, sorted=True) # NOTE: Make sure D2 is row-sorted as well
  D2, (ew, tw) = rips_boundary(D, p=2, diam=d, sorted=True) # keep sorted = True!
  if np.all(ew > b):
    return((0,0,0,0) if kwargs.get('summands', False) else 0)
  b_ind = 0 if np.all(ew > b) else np.max(np.flatnonzero(ew <= b))
  d_ind = 0 if np.all(tw > d) else np.max(np.flatnonzero(tw <= d))
  # if b_ind == 0 and d_ind == 0: 
  #   return((0,0,0) if kwargs.get('summands', False) else 0)
  i, j = b_ind+1, d_ind+1

  # if check_poset:
  #   from itertools import product, combinations
  #   Edges = np.reshape(D1.indices, (D1.shape[1], 2))
  #   Triangles = np.reshape(D2.indices, (D2.shape[1], 3))
  #   ## TODO: fix this 
  #   all_faces = [list(combinations(tri, 2)) for tri in Triangles[:j,:]]
  #   all_faces = np.array(all_faces).flatten()
  #   F = np.sort(np.unique(rank_combs(Edges[all_faces,:], n=X.shape[0], k=2)))
  #   # all_faces = np.unique(np.reshape(np.array(all_faces).flatten(), (len(all_faces)*3, 2)), axis=0)
  #   E = np.sort(rank_combs(Edges[:i,:], n=X.shape[0], k=2))
  #   # F = rank_combs(all_faces, n=X.shape[0], k=2)
  #   assert np.all(np.array([face in E for face in F]))
  return(persistent_betti(D1, D2, i, j, **kwargs))

def persistent_betti_rips_matrix(X: ArrayLike, b: float, d: float, p: int = 1, **kwargs):
  """ 
  X := point cloud or set of pairwise distances
  b := birth diameter threshold
  d := death diameter threshold 
  p := unused 
  kwargs 
  """
  assert is_point_cloud(X) or is_pairwise_distances(X), "Invalid format for rips"
  assert b <= d, "Birth time must be less than death time"
  D = pdist(X) if is_point_cloud(X) else X
  D1, (vw, ew) = rips_boundary(D, p=1, diam=d, sorted=True) # NOTE: Make sure D2 is row-sorted as well
  D2, (ew, tw) = rips_boundary(D, p=2, diam=d, sorted=True)
  if np.all(ew > b):
    T1, T2, T3 = csc_matrix((0,0)), csc_matrix((0,0)), csc_matrix((0,0))
    return(T1, T2, T3)
  D1.data = np.sign(D1.data)*np.repeat(ew, 2)
  D2.data = np.sign(D2.data)*np.repeat(tw, 3)
  b_ind = 0 if np.all(ew > b) else np.max(np.flatnonzero(ew <= b))
  d_ind = 0 if np.all(tw > d) else np.max(np.flatnonzero(tw <= d))
  i, j = int(b_ind+1), int(d_ind+1)
  T1 = D1[:,:i].copy()
  T2 = D2[:,:j].copy()
  T3 = D2[i:,:j].copy()
  return(T1, T2, T3)

def boundary_faces(D: csc_matrix):
  # D.eliminate_zeros()
  d = len(D[:,0].indices)
  F = np.reshape(D.indices, (D.shape[1], d))
  if d == 2:
    return(F)
  else: 
    all_faces = np.array([list(combinations(face, 2)) for face in F]).flatten()
  #   F = np.sort(np.unique(rank_combs(Edges[all_faces,:], n=X.shape[0], k=2)))

def rips_weights(X: ArrayLike, faces: ArrayLike):
  """ 
  X := point cloud or pairwise distances
  faces := (n x k) integer matrix of (k-1)-faces 
  """
  assert is_point_cloud(X) or is_pairwise_distances(X), "Invalid format for rips"
  XD = pdist(X) if is_point_cloud(X) else X
  n = inverse_choose(len(XD), 2)
  if faces.shape[1] == 2:
    w = XD[rank_combs(faces, n=n, k=2)]
  elif faces.shape[1] == 3:
    w = np.array([np.max([XD[rank_C2(T[0],T[1],n)], XD[rank_C2(T[0],T[2],n)], XD[rank_C2(T[1],T[2],n)]]) for T in faces])
  else: 
    w = np.array([np.max([XD[rank_C2(u,v,n)] for u,v in combinations(F, 2)]) for F in faces])
  return(w)

def rips_boundary(X: ArrayLike, p: int, diam: float = np.inf, sorted: bool = False, optimized: bool = True):
  """ 
  Computes the p-th boundary matrix of Rips complex directly from a point cloud or set of pairwise distances
  """
  # TODO: remove both steps and make specialized method based on distances alone
  assert is_point_cloud(X) or is_pairwise_distances(X), "Invalid format for rips"
  XD = pdist(X) if is_point_cloud(X) else X
  n = inverse_choose(len(XD), 2)
  if (optimized):
    if p == 0:
      D = csc_matrix([], shape=(0, 0))
      fw, cw = 0, np.zeros(n) # face, coface weights
    elif p == 1: 
      cdata, cindices, cindptr, ew = boundary.rips_boundary_1(XD, n, diam)
      D = csc_matrix((cdata, cindices, cindptr), shape=(n, len(cindptr)-1))
      if sorted: 
        eind = np.argsort(ew)
        D, ew = D[:,eind], ew[eind]
      fw, cw = np.zeros(n), ew
    elif p == 2:       
      cdata, cindices, cindptr, tw = boundary.rips_boundary_2(XD, n, diam)
      mask = np.cumsum(XD <= diam)
      cindices = (mask-1)[cindices] # re-map to lowest indices
      D = csc_matrix((cdata, cindices, cindptr), shape=(mask[-1], len(cindptr)-1))
      ew = XD[XD <= diam]
      if sorted: 
        eind, tind = np.argsort(ew), np.argsort(tw)
        D, ew, tw = D[np.ix_(eind,tind)], ew[eind], tw[tind]
      fw, cw = ew, tw
    return(D, (fw, cw)) # face, coface weights
  else:
    K = rips(XD, p=p, diam=diam)
    D = boundary_matrix(K, p=p)
    if p == 0:
      fw = 0
      cw = np.zeros(n)
    elif p == 1: 
      w = rips_weights(XD, K['edges'])
      # w = XD[rank_combs(K['edges'], n=n, k=2)]
      if sorted: 
        eind = np.argsort(w)
        D, w = D[:,eind], w[eind]
      fw, cw = np.zeros(n), w
    elif p == 2:
      #tri_weight = lambda T: np.max([XD[rank_C2(T[0],T[1],n)], XD[rank_C2(T[0],T[2],n)], XD[rank_C2(T[1],T[2],n)]])
      #ew = XD[rank_combs(K['edges'], n=n, k=2)]
      #tw = np.array([tri_weight(tri) for tri in K['triangles']])
      ew, tw = rips_weights(XD, K['edges']), rips_weights(XD, K['triangles'])
      if sorted: 
        eind, tind = np.argsort(ew), np.argsort(tw)
        D, fw, cw = D[np.ix_(eind, tind)], ew[eind], tw[tind]
      else: 
        fw, cw = ew, tw # face, coface weights
    return(D, (fw, cw)) # face, coface weights

def low_entry(D: lil_array, j: Optional[int] = None):
  """ 
  Provides fast access to all the low entries of a given matrix 'D'
  
  If D is CSC, low(D,j) takes O(1) time
  If 
  
  """
  #assert isinstance(D, csc_matrix)
  if j is None: 
    assert isinstance(D, spmatrix) or isinstance(D, np.ndarray)
    m = np.repeat(-1, D.shape[1])
    r,c = D.nonzero() if isinstance(D, spmatrix) else np.argwhere(D != 0).T
    np.maximum.at(m,c,r)
    return m
    # return(np.array([low_entry(D, j) for j in range(D.shape[1])], dtype=int))
  else: 
    if isinstance(D, csc_matrix):
      nnz_j = np.abs(D.indptr[j+1]-D.indptr[j]) 
      return(D.indices[D.indptr[j]+nnz_j-1] if nnz_j > 0 else -1)
    elif isinstance(D, spmatrix): 
      return(-1 if D[:,[j]].getnnz() == 0 else max(D[:,[j]].nonzero()[0]))
      # max(zip(*R[:,[1]].nonzero()), key=lambda rc: (rc[0], rc[1] != 0))
    elif isinstance(D, np.ndarray):
      return(-1 if all(D[:,j] == 0) else max(np.flatnonzero(D[:,j] != 0)))
    else: 
      raise ValueError(f"Invalid input type {type(D)}for D ")

def pHcol(R: lil_array, V: lil_array, I: Optional[Iterable] = None):
  # assert isinstance(R, csc_matrix) and isinstance(V, csc_matrix), "Invalid inputs"
  assert R.shape[1] == V.shape[0], "Must be matching boundary matrices"
  m = R.shape[1]
  I = range(1, m) if I is None else I
  piv = np.array([low_entry(R, j) for j in range(m)], dtype=int)
  for j in I:
    while( piv[j] != -1 and np.any(J := piv[:j] == piv[j]) ):
      i = np.flatnonzero(J)[0] # i < j
      c = R[piv[j],j] / R[piv[i],i]
      R[:,[j]] -= c*R[:,[i]] 
      V[:,[j]] -= c*V[:,[i]] 
      _perf['n_col_adds'] += 2
      # print(np.max(np.linalg.svd(R.A)[1]))
      if isinstance(R, csc_matrix): 
        R.eliminate_zeros() # needed to changes the indices
      # piv[j] = -1 if len(R[:,j].indices) == 0 else R[:,j].indices[-1] # update pivot array
      piv[j] = -1 if R[:,[j]].getnnz() == 0 else max(R[:,[j]].nonzero()[0]) # update pivot array
  return(None)



def reduction_pHcol(D1: lil_array, D2: lil_array, clearing: bool = False):
  # assert isinstance(D1, csc_matrix) and isinstance(D2, csc_matrix), "Invalid boundary matrices"
  assert D1.shape[1] == D2.shape[0], "Must be matching boundary matrices"
  V1, V2 = sps.identity(D1.shape[1]).tolil(), sps.identity(D2.shape[1]).tolil()
  R1, R2 = D1.copy(), D2.copy()
  if not(clearing):
    pHcol(R1, V1)
    pHcol(R2, V2)
  else: 
    pHcol(R2, V2)
    cleared = np.array(list(filter(lambda p: p != -1, low_entry(R2))))
    uncleared = np.setdiff1d(np.array(list(range(R1.shape[1]))), cleared)
    pHcol(R1, V1, uncleared)
    R1[:,cleared] = 0
    #R1.eliminate_zeros()
    cleared_domain = np.flatnonzero(low_entry(R2) != -1)
    V1[:,cleared] = R2[:,cleared_domain] # set Vi = Rj
  return((R1, R2, V1, V2))

def reduction_pHcol_clearing(D: Iterable[lil_array], implicit: bool = False):
  """ 
  D must be given as an iterable of boundary matrices given in decreasing dimensions
  """
  R, V = [], []
  cleared = np.array([], dtype=int)
  for Dp in D:
    m_p = Dp.shape[1] 
    Rp, Vp = Dp.copy(), sps.identity(m_p).tocsc()
    uncleared = np.array(list(filter(lambda p: not(p in cleared), range(m_p))))
    pHcol(Rp, Vp, uncleared) # is_reduced(Rp[:,uncleared]) == true
    R.append(Rp)
    V.append(Vp)
    cleared = np.array(list(filter(lambda p: p != -1, low_entry(Rp))))
  return((R, V))

def is_reduced(R: lil_array) -> bool:
  """ Checks if a matrix R is indeed reduced. """
  # assert isinstance(R, csc_matrix), "R must be a CSC sparse matrix."
  if isinstance(R, csc_matrix): 
    R.eliminate_zeros() # fix indices if need be
  low_ind = np.array([low_entry(R, j) for j in range(R.shape[1])], dtype=int)
  low_ind = low_ind[low_ind != -1]
  return(len(np.unique(low_ind)) == len(low_ind))

from pbsig.simplicial import FiltrationLike 

# def ph(K: FiltrationLike, p: int = 0, factor: bool = False):
#   """
#   Computes the p-th persistent homology a filtration 'K'. 

#   Returns: 
#     (1) dgm := The p-th persistence diagram of K (if factor == False), or;
#     (2) (Dp, Dq, Rp, Rq, Vp, Vq) := The matrices used to compute the decomposition R = D V 
#   """
#   D0, D1 = boundary_matrix(K, p=(0,1))
#   R0, R1, V0, V1 = reduction_pHcol(D0, D1)
#   Vf = dict(sorted(zip(to_str(V), fv0), key=index(1)))
#   Ef = dict(sorted(zip(to_str(E), fe0), key=index(1)))


# from enum import Enum
def projector_intersection(A, B, space: str = "RR", method: str = "neumann", eps: float = 100*np.finfo(float).resolution):
  """ 
  Creates a projector that projects onto the intersection of spaces (A,B), where
  A and B are linear subspace.
  
  method := one of 'neumann', 'anderson', or 'israel' 
  """
  assert space in ["RR", "RN", "NR", "NN"], "Invalid space argument"
  # if A.shape
  U, E, Vt = np.linalg.svd(A.A, full_matrices=False)
  X, S, Yt = np.linalg.svd(B.A, full_matrices=False)
  PA = U @ U.T if space[0] == 'R' else np.eye(A.shape[1]) - Vt.T @ Vt
  PB = X @ X.T if space[1] == 'R' else np.eye(B.shape[1]) - Yt.T @ Yt
  if method == "neumann":
    PI = np.linalg.matrix_power(PA @ PB, 125)# von neumanns identity
  elif method == "anderson":
    PI = 2*(PA @ np.linalg.pinv(PA + PB) @ PB)
  elif method == "israel":
    L, s, _ = np.linalg.svd(PA @ PB, full_matrices=False)
    s_ind = np.flatnonzero(abs(s - 1.0) <= eps)
    PI = np.zeros(shape=PA.shape)
    for si in s_ind:
      PI += L[:,[si]] @ L[:,[si]].T
  else:
    raise ValueError("Invalid method given.")
  return(PI)
  # np.sum(abs(np.linalg.svd(np.c_[PI @ D1.T, PI @ D2])[1]))

def plot_rips(X, diam: float, vertex_opt: Dict = {}, edge_opt: Dict = {}, poly_opt: Dict = {}, balls: bool = False, **kwargs):
  import matplotlib.pyplot as plt
  from matplotlib.patches import Polygon
  from matplotlib.collections import PatchCollection
  K = rips(X, diam=diam, p=2)
  fig = kwargs.pop('fig', None)
  ax = kwargs.pop('ax', None)
  if fig is None:
    fig = plt.figure(**kwargs)
  else: 
    fig = fig
  ax = ax if not(ax is None) else fig.gca()
  
  ## Vertices - draw in front
  ax.scatter(*X.T, zorder=20, **vertex_opt)

  ## Edges 
  for u,v in K['edges']:
    ax.plot(*X[(u,v),:].T, c=edge_opt.get('c', 'black'), zorder=10)
  
  ## Triangles 
  triangles = []
  for u,v,w in K['triangles']:
    polygon = Polygon(X[(u,v,w),:], True)
    triangles.append(polygon)
  poly_opt['alpha'] = poly_opt.get('alpha', 0.40)
  poly_opt['zorder'] = 0
  p = PatchCollection(triangles, **poly_opt)
  p.set_color(poly_opt.get('c', (255/255, 215/255, 0, 0)))
  ax.add_collection(p)  
  
  ## Optionally draw the balls to visualize rips/cech better
  if balls: 
    for x in X: 
      ball = plt.Circle(x, diam/2, color='y', clip_on=False) 
      ax.add_patch(ball)
  ax.set_aspect('equal')


# TODO: Remove names, include in f. That is, f is either a tuple of ArrayLike's or a tuple of Dict's 
def persistence_pairs(R1, R2: Optional[csc_matrix] = None, f: Sequence[Union[ArrayLike, Dict]] = None, collapse: bool = True):
  """
  R1 := reduced matrix whose pivot/zero columns mark the destroyers/creators of the p-1/p homology classes, with columns and rows in filtration order*
  R2 := reduced matrix whose pivot/zero columns mark the destroyers/creators of the p/p+1 homology classes, with columns and rows in filtration order*
  f := a 2-tuple (f0,f1) where each fx is either a sorted arraylike of filtration values or a dictionary mapping simplex names -> filtration values
  collapse := boolean indicating whether to collapse pairs with 0 persistence

  ** Note **:  
  Both the rows and columsn of R1 are assumed to be filtration order! That is, the p-chains (p-1 chains) corresponding to the columns (rows) of R1 
  must respect the poset and filtration order induced by the underlying filter function. 

  In contrast, if (f0,f1) are given as (simplex -> filtration value) key-value pairs, then said pairs need not necessarily be in any order. 
  However, then *lexicographic refinement* of the mapping will then be assumed to match the columns and rows of their respective reduced matrices. 

  The lexicographic refinement order of a given map M: K -> F mapping sorts simplices s \in K first by filtration value M(s), then by lexicographic order, i.e. 
  if f(s1) = f(s2) for s1,s2 \in K, then s1 < s2 in the lexicographic refinement is s1 is lexicographically smaller than s2. 

  """
  from operator import itemgetter
  assert is_reduced(R1), "Passed in matrix is not in reduced form"
  births_index = np.flatnonzero(low_entry(R1) == -1)
  deaths_index = low_entry(R2)
  
  ## Handle f, deduce simplex names, etc
  if f is None: 
    f = (np.array(range(R1.shape[1])), np.array(range(R2.shape[1])))
  assert len(f) == 2, "f must be tuple of function values"
  if isinstance(f[0], dict) and isinstance(f[1], dict):
    from pbsig.utility import is_sorted
    
    ## See the note
    f0 = f[0] if is_sorted(f[0].values()) else { k:v  for v,k in sorted(zip(f[0].values(), f[0].keys())) }
    f1 = f[1] if is_sorted(f[1].values()) else { k:v  for v,k in sorted(zip(f[1].values(), f[1].keys())) }
    ew, tw = np.fromiter(f0.values(), dtype=float), np.fromiter(f1.values(), dtype=float)
    assert is_sorted(ew) and is_sorted(tw)
    b_names, d_names = np.array(list(f0.keys())), np.array(list(f1.keys()))
    names = True
  else: 
    assert isinstance(f[0], np.ndarray) and isinstance(f[1], np.ndarray)
    ew, tw = f
    names = False

  ## Handle non-essential pairs 
  b_ind = deaths_index[deaths_index != -1]
  d_ind = np.flatnonzero(deaths_index != -1)
  births, deaths = ew[b_ind], tw[d_ind]
  dgm = np.c_[births, deaths]
  
  ## Retrieve birth/death names, if names given
  if names: s_names = np.c_[b_names[b_ind], d_names[d_ind]]

  ## Handle essential pairs 
  essential_ind = np.setdiff1d(births_index, deaths_index[deaths_index != -1])
  dgm = np.vstack((dgm, np.c_[ew[essential_ind], np.repeat(np.inf, len(essential_ind))]))
  if names: s_names = np.vstack((s_names, np.c_[b_names[essential_ind], np.repeat(type(b_names[0])(), len(essential_ind))]))

  ## Sort birth birth/death pairs + names for reproducibility
  lex_ind = np.lexsort(np.rot90(dgm))
  dgm = dgm[lex_ind,:]
  if names: s_names = s_names[lex_ind,:]

  ## Collapse zero-persistence pairs if required
  if collapse: 
    pos_pers = abs(dgm[:,1] - dgm[:,0]) >= 10*np.finfo(float).eps
    valid_ind = np.logical_or(dgm[:,1] == np.inf, pos_pers)
  else: 
    valid_ind = np.fromiter(range(dgm.shape[0]), dtype=int)
  
  ## Tack on names if requested
  if names:
    return({ 'creators' : s_names[valid_ind,0], 'destroyers': s_names[valid_ind,1], 'dgm': dgm[valid_ind,:] })
  else: 
    return(dgm[valid_ind,:])

def validate_decomp(D1, R1, V1, D2 = None, R2 = None, V2 = None, epsilon: float = 10*np.finfo(float).eps):
  valid = is_reduced(R1)
  valid &= np.isclose(sum(abs(((D1 @ V1) - R1).data).flatten()), 0.0)
  valid &= np.isclose(np.sum(V1 - triu(V1)), 0.0)
  if not(D2 is None):
    valid &= is_reduced(R2)
    valid &= np.isclose(sum(abs(((D2 @ V2) - R2).data).flatten()), 0.0) #np.isclose(np.sum((D2 @ V2) - R2), 0.0)
    valid &= np.isclose(np.sum(V2 - triu(V2)), 0.0)
  return(valid)


def generate_dgm(K: MutableFiltration, R: spmatrix, collapse: bool = True) -> ArrayLike :
  """ Returns the persistence diagram from (K, R) """
  rlow = low_entry(R)
  sdim = np.array([s.dim() for s in iter(K.values())])
  
  ## Get the indices of the creators and destroyers
  if any(rlow == -1):
    creator_mask = rlow == -1
    creator_dim = sdim[creator_mask]
    birth = np.flatnonzero(creator_mask)
    death = np.repeat(np.inf, len(birth))
    death[np.searchsorted(birth, rlow[~creator_mask])] = np.flatnonzero(~creator_mask)
  else:
    creator_dim = np.empty(0)
    birth = np.empty(shape=(0, 1))
    death = np.empty(shape=(0, 1))
  
  ## Trivial case
  if len(creator_dim) == 0: return dict()

  ## Match the barcodes with the index set of the filtration
  key_dtype = type(next(K.keys()))
  filter_vals = np.fromiter(K.keys(), dtype=key_dtype)
  index2f = {i:fv for i, fv in zip(np.arange(len(K)), filter_vals)} | { np.inf : np.inf}
  birth = np.array([index2f[i] for i in birth])
  death = np.array([index2f[i] for i in death])
  
  ## Assemble the diagram
  dgm = np.fromiter(zip(birth, death), dtype=[('birth', 'f4'), ('death', 'f4')])
  if collapse:
    nonzero_ind = ~np.isclose(dgm['death'] - dgm['birth'], 0.0) 
    dgm = dgm[nonzero_ind]
    creator_dim = creator_dim[nonzero_ind]

  ## Split diagram based on homology dimension
  dgm = { p : np.take(dgm, np.flatnonzero(creator_dim == p)) for p in np.sort(np.unique(creator_dim)) }
  return dgm 

## TODO: redo with filtration class at some point
def barcodes(K: MutableFiltration, p: Optional[int] = None, f: Tuple= None, **kwargs):
  """
  Given a filtered simplicial complex 'K' and optionally an integer p >= 0, generate p-dimensional barcodes

  If p is not specified, all p-dimensional barcodes are generated up to the dimension of the filtration.
  """
  assert isinstance(K, MutableFiltration), "Only accepts filtration objects for now"
  if p is None:
    D = boundary_matrix(K)
    V = sps.identity(D.shape[1]).tolil()
    R = D.copy().tolil()
    pHcol(R, V)
    assert validate_decomp(D, R, V)
    dgm = generate_dgm(K, R)      
    ## Generators 
    # V[np.ix_(creator_mask,creator_mask)]
    # [simplices[int(i)] for i in dgm_index[0][:,0]]
    # [simplices[int(i)] for i in dgm_index[0][:,1] if i != np.inf]
    # b = np.flatnonzero(rlow[rlow != -1])

    # assert validate_decomp(D0, R0, V0, D1, R1, V1)
  return(dgm)

# def persistent_pairs(R: List[csc_matrix], K: Dict, F: List[ArrayLike] = None, cohomology: bool = False):
# p_birth = np.where(low_entry(R1) == -1)
# p_death = 
#ind = np.setdiff1d(np.fromiter(range(Rp.shape[1]), dtype=int), cleared)
#low_p = np.array([low_entry(Rp, j) for j in range(Rp.shape[1])], dtype=int)
#cleared = low_p[low_p != -1]
# piv1 = np.array([low_entry(D1,j) for j in range(m)], dtype=int
## Reduce D1
# for j in range(1, m):
#   while( piv1[j] != -1 and np.any(J := piv1[:j] == piv1[j]) ):
#     i = np.flatnonzero(J)[0] # i < j
#     c = D1[piv1[j],j] / D1[piv1[i],i]
#     D1[:,j] -= c*D1[:,i] # real-valued case
#     V1[:,j] -= c*V1[:,i] # also do V1
#     D1.eliminate_zeros() # needed to changes the indices
#     piv1[j] = -1 if len(D1[:,j].indices) == 0 else D1[:,j].indices[-1]
## Reduce D2
# piv2 = np.array([low_entry(D2,j) for j in range(k)], dtype=int)
# for j in range(1, k):
#   while( piv2[j] != -1 and np.any(J := piv2[:j] == piv2[j]) ):
#     i = np.flatnonzero(J)[0] # i < j
#     c = D2[piv2[j],j] / D2[piv2[i],i]
#     D2[:,j] -= c*D2[:,i] # real-valued case
#     V2[:,j] -= c*V2[:,i] # also do V1
#     D2.eliminate_zeros() # needed to changes the indices
#     piv2[j] = -1 if len(D2[:,j].indices) == 0 else D2[:,j].indices[-1]

## Add persistent pairs
# I1 = D1.indices[D1.indptr[1]:D1.indptr[2]]
# I2 = D1.indices[D1.indptr[2]:D1.indptr[3]]
# np.bitwise_xor
# np.setxor1d(I1, I2)
# D1[:,2] = D1[:,2] + D1[:,3]
# raise ValueError("test")

# class Reduction():
#   def __init__(D0: csc_matrix, D1: csc_matrix): # boundary matrix


# def boundary_matrix(S: Iterable, F: Iterable):
#   """
#   Creates a p-boundary matrix (CSC format) from an iterable of simplices.

#   Assumes S only enumerates p-simplices.

#   Parameters:
#     S := Iterable over p-simplices
#     F := Iterable over (p-1)-faces
#   """ 
#   array('I')

def sw_parameters(bounds: Tuple, d: int = None, tau: float = None,  w: float = None, L: int = None):
  """ 
  Computes 'nice' slidding window parameters. 
  
  Given an interval [a,b] and any two slidding window parameters (d, w, tau, L), 
  returns the necessary parameters (d, tau) needed to construct Takens slidding window embedding 
  
  """
  if not(d is None) and not(tau is None):
    return(d, tau)
  elif not(d is None) and not(w is None):
    return(d, w/d)
  elif not(d is None) and not(L is None):
    w = bounds[1]*d/(L*(d+1))
    return(d, w/d)
  elif not(tau is None) and not(w is None):
    d = int(w/tau)
    return(d, tau)
  elif not(tau is None) and not(L is None):
    d = bounds[1]/(L*tau) - 1
    return(d, tau)
  elif not(w is None) and not(L is None):
    d = np.round(1/((bounds[1] - w*L)/(w*L)))
    assert d > 0, "window*L must be less than the entire time duration!"
    return(d, w/d)
  else:
    raise ValueError("Invalid combination of parameters given.")


def sliding_window(f: Union[ArrayLike, Callable], bounds: Tuple = (0, 1)):
  ''' 
  Slidding Window Embedding of a time series.
  
  Returns a function which generates a n-point slidding window point cloud of a fixed time-series/function _f_. 

  The returned function has the parameters
    - n := number of point to generate the embedding
    - d := dimension-1 of the resulting embedding 
    - w := (optional) window size each (d+1)-dimensional delay coordinate is derived from
    - tau := (optional) step size 
    - L := (optional) expected number of periods, if known
  The parameter n and d must be supplied, along with exactly one of 'w', 'tau' or 'L'.  
  '''
  assert isinstance(f, Callable) or isinstance(f, ArrayLike), "Time series must be function or array like"
  # Assume function like, defined on [0, 1]
  # Otherwise, construct a continuous interpolation via e.g. cubic spline
  def sw(n: int, d: int = None, tau: float = None, w: float = None, L: int = None):
    ''' Creates a slidding window point cloud over 'n' windows '''
    d, tau = sw_parameters(bounds, d=d, tau=tau, w=w, L=L)
    T = np.linspace(bounds[0], bounds[1] - d*tau, n)
    delay_coord = lambda t: np.array([f(t + di*tau) for di in range(d+1)])
    X = np.array([delay_coord(t) for t in T])
    return(X)
  return(sw)

## TODO: generalize beyond lower stars! 
# TODO: change E to be iterable, add flag to accept pre-sorted edge iterables
def ph0_lower_star(fv: ArrayLike, E: ArrayLike, collapse: bool = True, lex_sort: bool = True, max_death: Any = ["inf", "max"]) -> ArrayLike:
  """
  Computes the 0-dim persistence diagram of a lower-star filtration. 

  Parameters: 
    fv := 1d np.array of vertex function values, i.e. f(v_i) = fv[i] (index ordered)
    E := (m x 2) np.array of edges in the underlying complex
    collapse := whether to collapse 0-persistence pairs. Defaults to True. 
    lex_sort := whether to return the pairs in lexicographical birth-death order. Default to True.

  Return: 
    barcodes 
  """
  from scipy.cluster.hierarchy import DisjointSet
  if isinstance(max_death, list) and max_death == ["inf", "max"]:
    max_death = np.inf
  elif max_death == "max":
    max_death = max(fv)
  else:
    assert isinstance(max_death, float)

  ## Data structures + variables
  nv = len(fv)
  ds = DisjointSet(range(nv)) # Set representatives are minimum by *index*
  elders = np.fromiter(range(nv), dtype=int) # Elder map: needed to maintain minimum set representatives
  paired = np.zeros(nv, dtype=bool)
  insert_rule = lambda a,b: not(np.isclose(a,b)) if collapse else True # when to insert a pair
  
  ## Compute lower-star edge values; prepare to traverse in order
  ne = E.shape[0]
  ei = np.fromiter(sorted(range(ne), key=lambda i: (max(fv[E[i,:]]), min(fv[E[i,:]]))), dtype=int)
  
  ## Proceed to union components via elder rule
  dgm = []
  for (i,j), f in zip(E[ei,:], fv[E[ei,:]].max(axis=1)):
    ## The 'elder' was born first before the child: has smaller function value
    elder, child = (i, j) if fv[i] <= fv[j] else (j, i)
    if not ds.connected(i,j):
      if not paired[child]: # child unpaired => merged instantly by (i,j)
        dgm += [(fv[child], f)] if insert_rule(fv[child], f) else []
        paired[child] = True # kill the child
      else: # child already paired in component, use elder rule (keep elder alive)
        creator = elders[ds[elder]] if fv[elders[ds[child]]] <= fv[elders[ds[elder]]] else elders[ds[child]]
        dgm += [(fv[creator], f)] if insert_rule(fv[creator], f) else []
        paired[creator] = True
      
    ## Merge (i,j) and update elder map
    elder_i, elder_j = elders[ds[i]], elders[ds[j]]
    ds.merge(i,j)
    elders[ds[i]] = elders[ds[j]] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j

  ## Post-processing 
  eb = fv[np.flatnonzero(paired == False)]  ## essential births
  ep = list(zip(eb, [max_death]*len(eb)))   ## essential pairs 
  dgm = np.array(dgm + ep)                  ## append essential pairs

  ## If warranted, lexicographically sort output
  return dgm if not(lex_sort) else dgm[np.lexsort(np.rot90(dgm)),:]

def lower_star_ph_dionysus(f: ArrayLike, E: ArrayLike, T: ArrayLike):
  """
  Compute the p=(0,1) persistence diagrams of the lower-star filtration with vertex values 'f' and triangles 'T'
  """
  import dionysus as d
  #n = len(f)
  vertices = [([i], w) for i,w in enumerate(f)]
  edges = [(list(e), np.max(f[e])) for e in E]
  triangles = [(list(t), np.max(f[t])) for t in T]
  F = []
  for v in vertices: F.append(v)
  for e in edges: F.append(e) 
  for t in triangles: F.append(t)

  
  filtration = d.Filtration()
  for v_ind, time in F: filtration.append(d.Simplex(v_ind, time))
  filtration.sort()
  m = d.homology_persistence(filtration)
  dgms = d.init_diagrams(m, filtration)
  DGM0 = np.array([[pt.birth, pt.death] for pt in dgms[0]])
  DGM1 = np.array([[pt.birth, pt.death] for pt in dgms[1]])
  return([DGM0, DGM1])
