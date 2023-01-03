## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import *
from pbsig.utility import *

def generate_dataset(n: int = 15, d: int = 2):
  X = np.random.uniform(size=(n,d))
  K = delaunay_complex(X) 
  return X, K

def test_generate():
  X, K = generate_dataset(15)
  assert isinstance(X, np.ndarray)
  assert isinstance(K, SimplicialComplex)

def test_can_import():
  from pbsig.linalg import laplacian
  assert str(type(laplacian)) == "<class 'module'>"

def test_weighted_boundary_p1():
  X, S = generate_dataset()
  D1 = boundary_matrix(S, p=1)

  ## Ensure nice CSC matrix
  D1 = D1.tocsc()
  assert D1.has_sorted_indices, "Must be nice csc matrix"

  ## Generate weights 
  wp, wq = [np.random.uniform(size=d, low=0.0, high=1.0) for d in S.shape[:2]]
  assert all(wp > 0) and all(wq > 0)

  ## Make boundary matrix weighted
  W1 = D1.copy()
  vw = wp[W1.indices]
  ew = np.repeat(wq, 2)
  W1.data = np.sign(W1.data)*ew*vw

  ## Check equivalence of boundary and weighted boundary matrix-vector product 
  y = np.random.uniform(size=D1.shape[1])
  assert np.allclose(W1 @ y, (diags(wp) @ D1 @ diags(wq)) @ y)

  ## Check equivalence of boundary and laplacian matrix-vector product 
  x = np.random.uniform(size=D1.shape[0])
  assert np.allclose(W1 @ W1.T @ x, diags(wp) @ D1 @ diags(wq**2) @ D1.T @ diags(wp) @ x), "Matvec failed"
  assert is_symmetric(W1 @ W1.T)

  ## Check sqrt of eigen-spectrum of laplacian yields singular values of weighted boundary 
  LW = W1 @ W1.T
  ew = np.sort(np.sqrt(np.linalg.eigvalsh(LW.todense())))
  sw = np.sort(np.linalg.svd(W1.todense())[1])
  assert np.allclose(ew[1:], sw[1:])

  ## Check matrix-vector product sqrt
  W1 = D1.copy()
  vw = wp[W1.indices]
  ew = np.repeat(wq, 2)
  W1.data = np.sign(W1.data)*np.sqrt(ew)*np.sqrt(vw)
  assert np.allclose(W1 @ W1.T @ x, diags(np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(np.sqrt(wp)) @ x), "Matvec failed"
  assert is_symmetric(W1 @ W1.T)

  ## Test symmetry
  assert is_symmetric(diags(np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(np.sqrt(wp)))
  assert is_symmetric(diags(1/np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(wp)))
  
def test_uplaplacian0():
  X, S = generate_dataset()
  D1 = boundary_matrix(S, p=1).tocsc()
  wq = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
  wpl = np.random.uniform(size=S.shape[0], low=0.0, high=1.0)
  wpr = np.random.uniform(size=S.shape[0], low=0.0, high=1.0)

  ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
  x = np.random.uniform(size=D1.shape[0])
  LW = diags(wpl) @ D1 @ diags(wq) @ D1.T @ diags(wpr)

  ## Check diagonal first
  V, E = list(S.faces(0)), list(S.faces(1))
  d = np.zeros(S.shape[0])
  for s_ind, s in enumerate(E):
    for f in s.boundary():
      ii = V.index(f)
      d[ii] += wq[s_ind] * wpl[ii] * wpr[ii]
  assert np.allclose(d, LW.diagonal())

  ## Then check the full asymmetric product
  sgn_pattern = D1[:,[0]].data
  z = d * x # update with x
  for s_ind, s in enumerate(E):
    for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), -sgn_pattern):
      ii, jj = V.index(f1), V.index(f2)
      z[ii] += x[jj] * wpl[ii] * wpr[jj] * wq[s_ind] * sgn_ij
      z[jj] += x[ii] * wpl[jj] * wpr[ii] * wq[s_ind] * sgn_ij
  assert np.allclose(z, LW @ x)


def test_up_laplacian1():
  X, S = generate_dataset()
  D = boundary_matrix(S, p=2).tocsc()
  wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
  wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
  wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

  ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
  x = np.random.uniform(size=S.shape[1])
  LW = diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr)

  ## Check diagonal first
  P, Q = list(S.faces(1)), list(S.faces(2))
  d = np.zeros(S.shape[1])
  for s_ind, s in enumerate(Q):
    for f in s.boundary():
      ii = P.index(f)
      d[ii] += wpl[ii] * wq[s_ind] * wpr[ii]
  assert np.allclose(d, LW.diagonal())

  ## Then check the full asymmetric product
  sgn_pattern = D[:,[0]].data ## if vertex-induced
  z = d * x # update with x
  for s_ind, s in enumerate(Q):
    for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), -sgn_pattern):
      ii, jj = P.index(f1), P.index(f2)
      z[ii] += sgn_ij * x[jj] * wpl[ii] * wq[s_ind] * wpr[jj]
      z[jj] += sgn_ij * x[ii] * wpl[jj] * wq[s_ind] * wpr[ii]
  assert np.allclose(z, LW @ x)

## Test with arbitrary sign changes
def test_up_laplacian1_sgn():
  X, S = generate_dataset()
  D = boundary_matrix(S, p=2).tocsc()
  wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
  wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
  wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

  ## Orient arbitrarily
  for j in range(D.shape[1]):
    D[:,[j]] = D[:,[j]]*np.random.choice([-1,1])

  ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
  x = np.random.uniform(size=S.shape[1])
  LW = diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr)

  ## Check diagonal first
  P, Q = list(S.faces(1)), list(S.faces(2))
  d = np.zeros(S.shape[1])
  for s_ind, s in enumerate(Q):
    for f in s.boundary():
      ii = P.index(f)
      d[ii] += wpl[ii] * wq[s_ind] * wpr[ii]
  assert np.allclose(d, LW.diagonal())

  ## Then check the full asymmetric product
  z = d * x # update with x
  for s_ind, s in enumerate(Q):
    sgn_pattern = np.sign(D[:,[s_ind]].data)
    B = list(s.boundary())
    for (f1, f2) in combinations(s.boundary(), 2):
      sgn_ij = sgn_pattern[B.index(f1)]*sgn_pattern[B.index(f2)]
      ii, jj = P.index(f1), P.index(f2)
      z[ii] += sgn_ij * x[jj] * wpl[ii] * wq[s_ind] * wpr[jj]
      z[jj] += sgn_ij * x[ii] * wpl[jj] * wq[s_ind] * wpr[ii]
  assert np.allclose(z, LW @ x)

# https://stackoverflow.com/questions/38025074/how-to-accumulate-an-array-by-index-in-numpy
def test_vectorized_uplap():
  X, S = generate_dataset()
  L = up_laplacian(S, p=1, form='lo')

  x = np.random.uniform(size=L.shape[0])
  v = np.zeros(L.shape[0])
  v += L.degree * x.reshape(-1)
  q = len(L.simplices[0])-1
  N = 2*len(L.simplices)*comb(q+1, 2)
  P = np.zeros(shape=N, dtype=[('weights', 'f4'), ('xi', 'i2'), ('vo', 'i2')])
  cc = 0 
  for s_ind, s in enumerate(L.simplices):
    for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), L.sgn_pattern):
      ii, jj = L.index(f1), L.index(f2)
      d1 = sgn_ij * L._wfl[ii] * L._ws[s_ind] * L._wfr[jj]
      d2 = sgn_ij * L._wfl[jj] * L._ws[s_ind] * L._wfr[ii]
      P[cc] = d1, jj, ii
      P[cc+1] = d2, ii, jj
      cc += 2
  np.add.at(v, P['vo'], x[P['xi']]*P['weights'])
  assert np.allclose(L @ x, v)

  ## Ensure this works at library level 
  L = up_laplacian(S, p=1, form='lo')
  L.precompute()
  assert np.allclose(L._matvec(x), L._matvec_precompute(x))


  # X, S = generate_dataset(n=350)
  # L = up_laplacian(S, p=1, form='lo')

  # x = np.random.uniform(size=L.shape[0])
  # import timeit
  # timeit.timeit(lambda: L @ x, number=50) # 0.740328342999419

  # L.precompute()
  # timeit.timeit(lambda: L @ x, number=50) # 0.006494699999166187
  # L._matvec = L._matvec_precompute
  # #assert L._matvec(x) type error 
  # import types 
  # types.MethodType(L._matvec_precompute, L)



def test_up_laplacian1():
  X, S = generate_dataset()
  D = boundary_matrix(S, p=2).tocsc()
  wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
  wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
  wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

  from pbsig.linalg import UpLaplacian
  L = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
  
  ## Ensure construction
  from scipy.sparse.linalg import LinearOperator
  assert isinstance(L, LinearOperator)
  
  ## Test unweighted product 
  x = np.random.uniform(size=L.shape[0])
  assert np.allclose(D @ D.T @ x, L @ x)
  
  ## Test weighted product with p+1 simplices
  L.simplex_weights = wq 
  assert np.allclose(D @ diags(wq) @ D.T @ x, L @ x)

  ## Test weighted product with p simplices 
  L.face_right_weights = wpr 
  assert np.allclose(D @ diags(wq) @ D.T @ diags(wpr) @ x, L @ x)    
  L.face_left_weights = wpl 
  assert np.allclose(diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr) @ x, L @ x)  

  ## Test matrix equivalence at API level 
  from pbsig.linalg import up_laplacian
  LO = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
  LM = up_laplacian(S, p=1)
  assert np.allclose(LO @ x, LM @ x)

  ## Test matrix equivalence at API level 
  W = dict(zip(S, np.random.uniform(size=len(S), low=0.0)))
  LM = up_laplacian(S, p=1, weight = lambda s: W[s])
  LO = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
  LO.simplex_weights = np.array([W[s] for s in S.faces(2)])
  LO.face_left_weights = np.array([W[s] for s in S.faces(1)])
  LO.face_right_weights = np.array([W[s] for s in S.faces(1)])
  assert np.allclose(LO @ x, LM @ x)

  ## Test equivalence via weight function
  LO = up_laplacian(S, p=1, weight = lambda s: W[s], form='lo')
  assert np.allclose(LO @ x, LM @ x)

# LG = D1 @ diags(wq) @ D1.T 
# L_inner = diags(1/np.sqrt(wp)) @ LG @ diags(1/np.sqrt(wp))
# L_full = diags(np.sqrt(wp)) @ L_inner @ diags(1/np.sqrt(wp))
# assert is_symmetric(L_inner)
# assert not(is_symmetric(L_full))
# assert np.allclose(abs((LG @ diags(1/wp)) - L_full).data, 0.0)

# L_inner = diags(np.sqrt(wp)) @ LG @ diags(np.sqrt(wp))
# L_full = diags(1/np.sqrt(wp)) @ L_inner @ diags(np.sqrt(wp))
# assert is_symmetric(L_inner)
# assert not(is_symmetric(L_full))
# assert np.allclose(abs((LG @ diags(wp)) - L_full).data, 0.0)


def test_spectra_union():
  ## Test what happens with appropriate f respecting face poset with disjoint components
  S = SimplicialComplex([[1,2,3], [4,5,6]])
  D1 = boundary_matrix(S, p = 1)

  ## positive weights => same sparsity pattern
  wp = np.random.uniform(size=D1.shape[0])
  plt.spy(D1 @ D1.T @ diags(wp))

  ## Setting vertex weight to 0 makes a column 0 
  wp[2] = 0
  plt.spy(D1 @ D1.T @ diags(wp))

  ## Multiplying both sides sets row and column to 0 
  plt.spy(diags(wp) @ D1 @ D1.T @ diags(wp))
  
  ## Does not produce a block diagonal, but perhaps same rank....
  S = SimplicialComplex([[1,2,3,4,5,6]])
  D1 = boundary_matrix(S, p = 1)
  plt.spy(diags(wp) @ D1 @ D1.T @ diags(wp))

  wp = np.arange(1, D1.shape[0]+1)
  np.linalg.svd((D1 @ D1.T).todense())[1]
  np.linalg.svd((D1 @ D1.T @ diags(wp)).todense())[1]
  plt.spy(D1 @ D1.T @ diags(wp))

  np.allclose((LG @ diags(1/wp)).data, L_full.data)




  ## Test what happens when a vertex is zero'ed 
  from scipy.sparse.linalg import svds
  wv = wp.copy()
  piv = int(np.median(range(len(wv))))
  wv[piv] = 0.0
  LG = D1 @ diags(wq) @ D1.T
  sw_whole = svds(diags(np.sqrt(wp)) @ LG @ diags(np.sqrt(wp)), k = len(wv)-1)[1]
  plt.spy(diags(np.sqrt(wp)) @ LG @ diags(np.sqrt(wp)))
  
  LG_pos = diags(np.sqrt(wv)) @ LG @ diags(np.sqrt(wv))
  sw_pos = svds(LG_pos, k = len(wv)-1)[1]
  plt.spy(LG_pos)
  plt.spy(LG_pos[:piv,:piv])
  plt.spy(LG_pos[(piv+1):,(piv+1):])
  sw_tl = svds(LG_pos[:piv,:piv], k=piv-1)[1]
  sw_br = svds(LG_pos[(piv+1):,(piv+1):], k=piv-1)[1]
  
  total = np.array([l*r for l,r in product(sw_tl, sw_br)])
  np.union1d(sw_tl, sw_br)

  ## Try disjoint sets 
  S = SimplicialComplex([[1,2,3], [4,5,6]])
  D1 = boundary_matrix(S, p = 1)
  plt.spy(D1 @ D1.T)

  plt.spy(diags(1/np.sqrt(wv)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(wv)))



def test_weighted_up_laplacian():
  """ 
  This tests the matrix-vector product and quadratic forms of the pth up-Laplacian Lp = Dq @ Wq @ Dq.T @ Wp
  where q = p+1, Dq is the q-th boundary matrix, and (Wp,Wq) are diagonal matrices with *non-negative* entries.
  """
  X, S = generate_dataset()
  D1, D2 = boundary_matrix(S, p=(1,2))
  
  ## Generate weights 
  wp, wq = [np.random.uniform(size=d, low=0.0, high=1.0) for d in S.shape[:2]]

  ## Start with the graph Laplacian 
  L = D1 @ diags(wq) @ D1.T
  L_up = L @ diags(wp)
  L_up2 = diags(np.sqrt(wp)) @ L @ diags(np.sqrt(wp))
  x = np.random.uniform(size=L_up.shape[0])
  y = L_up @ x

  ## Test degree pre-computation first
  p_faces, q_faces = list(S.faces(p=0)), list(S.faces(p=1))
  z = np.zeros(S.shape[0]) 
  for q_ind, q in enumerate(q_faces):
    for p in boundary(q):
      p_ind = p_faces.index(p)
      z[p_ind] += wq[q_ind]**2 * wp[p_ind]**2
  assert max(abs(L_up.diagonal() - z)) < 1e-13

  x = np.random.uniform(size=ns[1])
  y = np.zeros(ns[1])
  del_sgn_pattern = [1,-1,1]
  lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(del_sgn_pattern,2)])
  for t_ind, t in enumerate(K.faces(p=2)):
    for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
      ii, jj = p_faces.index(e1), p_faces.index(e2)
      c = w0[ii] * w0[jj] * w1[t_ind]**2
      y[ii] += x[jj] * c * s_ij
      y[jj] += x[ii] * c * s_ij






def test_up_laplacian_native():
  from pbsig.linalg import laplacian
  X, K = generate_dataset(15)
  E = np.array(list(K.faces(1)))
  nv = K.dim()[0]
  er = rank_combs(E, k=2, n=nv)
  assert isinstance(er, np.ndarray) and len(er) == K.dim()[1]

  L = laplacian.UpLaplacian0_LS(er, nv)
  L.fv = np.random.uniform(size=nv)
  x = np.random.uniform(size=nv)
  L.prepare()
  L._matvec(x)

  from pbsig.linalg import up_laplacian
  fv = np.array(L.fv.copy())
  L_up = up_laplacian(K, p=0, weight=lambda s: max(fv[s]) if len(s) == 2 else 1.0)
  L_up @ x

def test_weighted_linear_L0():
  ## Generate random geometric complex
  X = np.random.uniform(size=(15,2))
  K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
  D1, D2 = boundary_matrix(K, p=(1,2))

  ## Test weighted linear form (only takes one O(n) vector!)
  ns = K.dim()
  w0, w1 = np.random.uniform(size=ns[0]), np.random.uniform(size=ns[1])
  v0 = np.zeros(ns[0]) # note the O(n) memory
  for cc, (i,j) in enumerate(K.faces(1)):
    v0[i] += w0[i]**2 * w1[cc]**2
    v0[j] += w0[j]**2 * w1[cc]**2
  v0 *= x # O(n)
  for cc, (i,j) in enumerate(K.faces(1)):
    c = w0[i]*w0[j]*w1[cc]**2
    v0[i] -= x[j]*c
    v0[j] -= x[i]*c
  v1 = (diags(w0) @ D1 @ diags(w1**2) @ D1.T @ diags(w0)) @ x
  assert np.allclose(v0, v1), "Unweighted Linear form failed!"

  ## Test interface method 
  from pbsig.linalg import up_laplacian
  fv = w0
  L_up = up_laplacian(K, p=0, weight=lambda s: max(fv[s]), form='lo')
  L_up.simplex_weights
  L_up._ws
  assert np.allclose(L_up @ x, v0)
  assert np.allclose(L_up @ x, v1)
  

def test_unsym_L_up():
  ## Generate random geometric complex
  X = np.random.uniform(size=(15,2))
  K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
  D1, D2 = boundary_matrix(K, p=(1,2))
  
  ns = K.dim()
  w0, w1 = np.random.uniform(size=ns[0], low=0.0, high=1.0), np.random.uniform(size=ns[1], low=0.0, high=1.0)
  assert all(w0 != 0.0) and all(w1 != 0.0)
  
  from scipy.sparse import diags
  from pbsig.linalg import is_symmetric
  assert is_symmetric(D1 @ D1.T), "unweighted 0-up-laplacian not symmetric!"
  assert is_symmetric(D1 @ diags(w1) @ D1.T), "edge weighted 0-up-laplacian not symmetric!"

  ## Start with just edge-weighted values
  L_inner = D1 @ diags(w1) @ D1.T
  ew_inner = np.linalg.eigvalsh(L_inner.todense())

  ## The inner-eigenvalues are identical to the non-symmetric v/e eigenvalues 
  LW_up_unsym = diags(1/np.sqrt(w0)) @ L_inner @ diags(np.sqrt(w0))
  ew_unsym = np.linalg.eig(LW_up_unsym.todense())[0]
  assert np.allclose(np.sort(ew_inner), np.sort(ew_unsym)) # the main statement
  assert np.allclose(np.sort(ew_inner), (1/np.sqrt(w0)) * ew_inner * np.sqrt(w0)) # same statement basically

  ## But this doesn't hold for the symmetric ones
  LW_up_sym = diags(np.sqrt(w0)) @ L_inner @ diags(np.sqrt(w0))
  ew_sym = np.linalg.eigh(LW_up_sym.todense())[0]
  assert np.allclose(np.sqrt(w0) * ew_sym * np.sqrt(w0), w0 * ew_sym)
  assert np.allclose(np.sqrt(w0) * ew_inner * np.sqrt(w0), w0 * ew_inner)
  assert np.allclose(np.sort(ew_inner * w0), np.sort(ew_sym))
  np.sort(ew_unsym)
  np.sort(ew_sym)
  np.sort((1/np.sqrt(w0)) * ew_inner * np.sqrt(w0))

  # np.linalg.eig( @ D1 @ diags(w1) @ D1.T @ diags(np.sqrt(w0))).todense())[0]
