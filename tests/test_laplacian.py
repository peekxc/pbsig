## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import *
from pbsig.utility import *
from pbsig.linalg import *

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
  L_up_asym = L @ diags(wp)
  L_up_sym = diags(np.sqrt(wp)) @ L @ diags(np.sqrt(wp))
  x = np.random.uniform(size=L_up.shape[0])
  y = L_up @ x

  ## Test degree pre-computation first
  p_faces, q_faces = list(S.faces(p=0)), list(S.faces(p=1))
  z = np.zeros(S.shape[0]) 
  for q_ind, q in enumerate(q_faces):
    for p in boundary(q):
      p_ind = p_faces.index(p)
      z[p_ind] += wq[q_ind]
  assert max(abs(L.diagonal() - z)) < 1e-13

  ## TODO: test inner product structure
  # x = np.random.uniform(size=ns[1])
  # y = np.zeros(ns[1])
  # del_sgn_pattern = [1,-1,1]
  # lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(del_sgn_pattern,2)])
  # for t_ind, t in enumerate(K.faces(p=2)):
  #   for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
  #     ii, jj = p_faces.index(e1), p_faces.index(e2)
  #     c = w0[ii] * w0[jj] * w1[t_ind]**2
  #     y[ii] += x[jj] * c * s_ij
  #     y[jj] += x[ii] * c * s_ij


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


def test_laplacian0_cpp():
  from pbsig.combinatorial import rank_combs
  X = np.random.uniform(size=(15,2))
  S = delaunay_complex(X) 
  r = rank_combs(S.faces(1), k=2, n=S.shape[0])
  x = np.random.uniform(size=L.shape[0])

  from pbsig.linalg import laplacian, up_laplacian
  L = laplacian.UpLaplacian0(r, X.shape[0], X.shape[0])
  L.compute_indexes()
  L.precompute_degree()
  LM = up_laplacian(S, p=0)
  assert np.allclose(L._matvec(x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)

  LO = up_laplacian(S, p=0, form='lo')
  assert np.allclose((LO @ x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)

  import timeit
  timeit.timeit(lambda: LM @ x, setup="import numpy as np; x = np.random.uniform(size=150)", number=1000)
  timeit.timeit(lambda: LO @ x, setup="import numpy as np; x = np.random.uniform(size=150)", number=1000)

def test_laplacian1_cpp():
  from pbsig.combinatorial import rank_combs
  X = np.random.uniform(size=(15,2))
  S = delaunay_complex(X) 
  qr = rank_combs(S.faces(2), k=3, n=S.shape[0])

  from pbsig.linalg import laplacian, up_laplacian
  L = laplacian.UpLaplacian1(qr, S.shape[0], S.shape[1])
  L.compute_indexes()
  L.precompute_degree()
  
  ## Ensure index map is valid
  pr_true = rank_combs(S.faces(1), k=2, n=S.shape[0])
  pr_test = np.sort(list(L.index_map.keys()))
  assert len(pr_true) == len(pr_test)
  assert all((pr_true - pr_test) == 0)

  ## Check matvec product 
  x = np.random.uniform(size=S.shape[1])
  LM = up_laplacian(S, p=1)
  assert np.allclose(L._matvec(x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)

def test_laplacian0_cpp_matmat():
  from pbsig.combinatorial import rank_combs
  X = np.random.uniform(size=(15,2))
  S = delaunay_complex(X) 
  r = rank_combs(S.faces(1), k=2, n=S.shape[0])

  from pbsig.linalg import laplacian, up_laplacian
  L = laplacian.UpLaplacian0(r, X.shape[0], X.shape[0])
  L.compute_indexes()
  L.precompute_degree()

  LM = up_laplacian(S, p=0)
  X = np.random.uniform(size=L.shape)
  assert np.allclose((LM @ X) - L._matmat(X), 0.0, atol=10*np.finfo(np.float32).eps)
  
  #L.fpr = np.random.uniform(size=S.shape[0])

def test_laplacian_API():
  from pbsig.linalg import laplacian, up_laplacian
  X = np.random.uniform(size=(150,2))
  S = delaunay_complex(X) 
  
  LM = up_laplacian(S, p=0, form='array')
  LO = up_laplacian(S, p=0, form='lo')
  x = np.random.uniform(size=S.shape[0])
  assert np.allclose(LM @ x, LO @ x, atol=10*np.finfo(np.float32).eps)

  LM = up_laplacian(S, p=1, form='array')
  LO = up_laplacian(S, p=1, form='lo')
  x = np.random.uniform(size=S.shape[1])
  assert np.allclose(LM @ x, LO @ x, atol=10*np.finfo(np.float32).eps)

  import timeit
  timeit.timeit(lambda: LM @ x, number=1000)
  timeit.timeit(lambda: LO @ x, number=1000)
  timeit.timeit(lambda: LO.L._matvec(x), number=1000)
  # LO.L._matvec(x)

# def test_weighted_linear_L0():
#   ## Generate random geometric complex
#   X = np.random.uniform(size=(15,2))
#   K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
#   D1, D2 = boundary_matrix(K, p=(1,2))

#   ## Test weighted linear form (only takes one O(n) vector!)
#   ns = K.dim()
#   w0, w1 = np.random.uniform(size=ns[0]), np.random.uniform(size=ns[1])
#   v0 = np.zeros(ns[0]) # note the O(n) memory
#   for cc, (i,j) in enumerate(K.faces(1)):
#     v0[i] += w0[i]**2 * w1[cc]**2
#     v0[j] += w0[j]**2 * w1[cc]**2
#   v0 *= x # O(n)
#   for cc, (i,j) in enumerate(K.faces(1)):
#     c = w0[i]*w0[j]*w1[cc]**2
#     v0[i] -= x[j]*c
#     v0[j] -= x[i]*c
#   v1 = (diags(w0) @ D1 @ diags(w1**2) @ D1.T @ diags(w0)) @ x
#   assert np.allclose(v0, v1), "Unweighted Linear form failed!"

#   ## Test interface method 
#   from pbsig.linalg import up_laplacian
#   fv = w0
#   L_up = up_laplacian(K, p=0, weight=lambda s: max(fv[s]), form='lo')
#   L_up.simplex_weights
#   L_up._ws
#   assert np.allclose(L_up @ x, v0)
#   assert np.allclose(L_up @ x, v1)


def test_laplacian_eigsh():
  LM = up_laplacian(S, p=0, form='array')
  solver = parameterize_solver(LM, solver='jd')
  ew_m = solver(LM)

  LO = up_laplacian(S, p=0, form='lo')
  solver = parameterize_solver(LO, solver='jd')
  ew_o = solver(LO)

  ## Ensure eigenvalues are all the same
  assert np.allclose(ew_o, ew_m, atol=1e-8)
  