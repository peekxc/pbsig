## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags, spmatrix
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import *
from pbsig.utility import *
from pbsig.linalg import *
from splex import simplicial_complex
from splex import ComplexLike
def generate_dataset(n: int = 15, d: int = 2):
  X = np.random.uniform(size=(n,d))
  K = delaunay_complex(X) 
  return X, K

def test_generate():
  X, K = generate_dataset(15)
  assert isinstance(X, np.ndarray)
  assert isinstance(K, ComplexLike)

def test_laplacian_op_0():
  S = simplicial_complex([[0,1],[0,2],[2,3]])
  L = up_laplacian(S, p=0, form='lo')
  assert np.all(L.simplices == np.array([[0,1], [0,2], [2,3]], dtype=np.int32))
  assert L.precompute_degree() is None
  assert all(L.faces == np.c_[np.array([0,1,2,3])])
  assert all(L._matvec(np.arange(L.shape[0])) == np.array([-3,1,1,1]))
  assert all(L._matvec(np.ones(L.shape[0])) == 0.0)
  LM = up_laplacian(S, p=0, form='array')
  assert isinstance(LM, spmatrix)
  assert LM.shape == tuple(L.shape)
  x = np.random.uniform(size=L.shape[0])
  assert np.allclose(LM @ x - L @ x, 0.0)

def test_laplacian_op_api():
  X, S = generate_dataset(30, 2)
  for p in range(0, 2):
    LO = up_laplacian(S, p=p, form='lo')
    assert set([Simplex(s) for s in LO.simplices]) == set(faces(S, p+1))
    assert set([Simplex(s) for s in LO.faces]) == set(faces(S, p))
    LM = up_laplacian(S, p=p, form='array')
    assert isinstance(LM, spmatrix)
    assert LM.shape == tuple(LO.shape)
    #x = np.random.uniform(size=L.shape[0])
    
    ## Ensure the degree are correct / ordering 
    ## use indicator vectors to assert equlity between matrix rep's
    n = card(S, p)
    for i in range(n):
      cv = np.array([0]*i + [1] + [0]*(n-i-1))
      assert np.isclose(np.max(abs(LM.todense()[i,:] - (LO @ cv))), 0)

    ## Ensure matvec is correct
    x = np.arange(card(S, p))
    assert np.allclose(LM @ x - LO @ x, 0.0)

def test_stability():
  S = simplicial_complex([[0,1,2], [0,1,3]])
  fv = np.array([1e-15, 0.2, 0.4, 1.5])
  # fv = np.array([1,1,1,1])
  fe = np.array([lower_star_weight(fv)(e) for e in faces(S, 1)])
  L1 = up_laplacian(S, p=0, normed=True, weight=lower_star_weight(fv))
  solver = eigvalsh_solver(L1)
  ew = solver(L1)

  def additive_noise(x: np.ndarray, eps: float = 1e-5):
    noise = np.random.uniform(size=len(x), low=-eps/2, high=eps/2)
    return x+noise

  def relative_noise(x: np.ndarray, eps: float = 1e-5):
    noise = np.random.uniform(size=len(x), low=-eps/2, high=eps/2)
    return x*(np.ones(len(x))+noise)

  ## Small absolute perturbation, potentially large relative
  # fv + np.random.uniform(size=len(fv), )
  # fv = np.maximum(additive_noise(fv, 1e-2), 0.0)
  fv = np.maximum(relative_noise(fv, 1e-2), 0.0) ## this should be regarded as a large perturbation

  # fv += np.array([-1e-15, 1e-15, 1e-15, 1e-15])
  L2 = up_laplacian(S, p=0, normed=True, weight=lower_star_weight(fv))
  ew1 = solver(L1)
  ew2 = solver(L2)
  print(f"Absolute error: {max(abs(ew2-ew1)):.12f}")
  print(f"Relative error: {max(abs(ew2-ew1)/(abs(ew1))):.12f}")

  chi_dist = lambda a,b: abs(np.sort(a)[::-1] - np.sort(b)[::-1])/(np.sqrt(abs(np.sort(a)[::-1] * np.sort(b)[::-1])))
  del_dist = lambda a,b: abs(np.sort(a)[::-1] - np.sort(b)[::-1])/(np.sqrt(abs(np.sort(a)[::-1]) + abs(np.sort(b)[::-1])))

  _, ua = np.linalg.eigh(L1.todense())
  _, ub = np.linalg.eigh(L2.todense())
  Q = np.dot(ua, ub.T) # this one
  print(f"L1: {np.around(L1.todense(), 5)}")
  print(f"L1 ew: {ew1}\n")
  print(f"L2: {np.around(L2.todense(), 5)}")
  print(f"L2 ew: {ew2}\n")
  print(f"Q: {np.around(Q, 5)} ")
  print(f"Q ew: {np.linalg.eigvalsh(Q)}\n")
  print(f"Q.T @ L1 @ Q: {np.around(Q.T @ L1 @ Q, 5)} \n")
  # print(f"Two sided error: {np.linalg.norm(L1 @ Q - L2 @ Q)}")
  print(f"Trace: {np.trace(Q.T @ L1 @ Q - L2)}, Fro: {np.linalg.norm(Q.T @ L1 @ Q - L2.todense())}")
  # np.around(u_opt.T @ L1 @ u_opt - L2.todense(), 5)
  print(f"Ostrowski Bound: {max(abs(np.linalg.eigvalsh(np.eye(4) - Q.T @ Q)))}")
  print(f"Max chi dist: {max(chi_dist(ew1,ew2))}, Max del dist: {max(del_dist(ew1,ew2))}")

  # from scipy.linalg import sqrtm
  # sqrtm()
  ew1, ev1 = np.linalg.eigh(L1.todense())
  ew2, ev2 = np.linalg.eigh(L2.todense())
  ew1[np.isclose(ew1, 0)] = 0
  ew2[np.isclose(ew2, 0)] = 0

  # A_isqrt = (ev1 @ np.diag(np.sqrt(pseudoinverse(ew1))) @ ev1.T)
  # A_sqrt = (ev1 @ np.diag(np.sqrt(ew1)) @ ev1.T)
  # B = A_isqrt @ L2.todense() @ A_isqrt
  # u,s,vt = np.linalg.svd(B)
  # Q = A_sqrt @ u @ vt @ A_sqrt
  # np.linalg.norm(Q.T @ L1 @ Q - L2)
  from scipy.optimize import minimize
  
  def obj(a):
    R = np.diag(np.sqrt(a)) @ Q @ np.diag(np.sqrt(a))
    return np.linalg.norm(R.T @ L1 @ R - L2.todense())
  xs = minimize(obj, x0 = np.ones(4)).x
  R = np.diag(np.sqrt(xs)) @ Q @ np.diag(np.sqrt(xs))
  print(np.linalg.norm(R.T @ L1 @ R - L2.todense()))
  print(np.sqrt(np.sum((ew1-ew2)**2)))
  # np.sqrt(np.sum((ew1 - ew2)**2))
  max(abs(np.linalg.eigvalsh(np.eye(4)-Q.T @ Q)))

  print(L1.todense())
  print(L2.todense())


  np.linalg.norm(L2.todense())
  
  L1.todense() - L2.todense()
  D1 = boundary_matrix(S, p = 1).todense()


def benchmark_matvec():
  import timeit
  X, S = generate_dataset(10000, 2)
  p = 1
  fv = np.random.uniform(size=card(S,0), low=0, high=5)
  LO = up_laplacian(S, p=p, form='lo', weight=lower_star_weight(fv))
  LM = up_laplacian(S, p=p, form='array', weight=lower_star_weight(fv))
  x = np.random.uniform(size=card(S,p), low=-1, high=1)
  timeit.timeit(lambda: LO @ x, number=1000)
  timeit.timeit(lambda: LM @ x, number=1000)
  assert max(abs((LM @ x) - (LO @ x))) <= 1e-5
  
## Test elementwise definitions
# D1 = boundary_matrix(S, p=1)
# fv = np.random.uniform(size=card(S,0), low=0, high=1)
# fe = np.array([lower_star_weight(fv)(e) for e in faces(S,1)])
# F = dict(zip(faces(S,1), fe))
# W0 = np.diag(fv)
# W1 = np.diag(fe)
# deg = np.zeros(card(S,0))
# for v in faces(S,0):
#   for e in S.cofaces(v):
#     if len(e) == len(v) + 1:
#       deg[v] += F[e]

# ## Correct 
# (F[(0,1)] + F[(0,2)])*fv[0]
# np.sqrt(W0) @ D1 @ W1 @ D1.T @ np.sqrt(W0)

# ## Correct 
# (F[(0,1)] + F[(0,2)])*deg[0]
# np.diag(np.sqrt(deg)) @ D1 @ W1 @ D1.T @ np.diag(np.sqrt(deg))

# ## Indeed: the diagonal is all ones 
# np.diag(1/np.sqrt(deg)) @ D1 @ W1 @ D1.T @ np.diag(1/np.sqrt(deg))

# ## Indeed: this matches
# up_laplacian(S, weight=lower_star_weight(fv), normed=True).todense()

# ## Matches higher order too 
# X, S = generate_dataset(30, 2)
# fv = np.random.uniform(size=card(S,0), low=0, high=5.0)
# np.diag(up_laplacian(S, weight=lower_star_weight(fv), normed=True, p=1).todense())
# L = up_laplacian(S, weight=lower_star_weight(fv), normed=True, p=1).todense()
# max(np.linalg.eigvalsh(L)) <= 3 # true

# L2 = up_laplacian(S, weight=lower_star_weight(150*fv), normed=True, p=1).todense()
# max(abs(np.linalg.eigvalsh(L2) - np.linalg.eigvalsh(L)))

# for a,b in zip(np.random.uniform(size=130, low=0, high=2), np.random.uniform(size=130, low=0, high=2)):
#   ub = abs(1/a - 1/b)
#   lb = abs(a-b)*(a*b)**(-1)
#   print(ub - lb)
#   # if not((ub - lb) > -1e-12):
#   #   print(a,b)


# def test_():
#   X, K = generate_dataset(5, d=2)
#   eps = 0.001
#   fv = np.random.uniform(size=card(K,0), low=0.0, high=5.0)
#   fv2 = fv + np.random.uniform(size=len(fv), low=-eps, high=eps)
#   actual_delta = max(abs(fv - fv2))
#   L = up_laplacian(K, p = 0, weight=lower_star_weight(fv))
#   L2 = up_laplacian(K, p = 0, weight=lower_star_weight(fv2))
  
#   ev = np.array([lower_star_weight(fv)(e) for e in faces(K, 1)])
#   D1 = boundary_matrix(K, p=1)
#   L = np.diag(1/np.sqrt(fv)) @ D1 @ np.diag(ev) @ D1.T @ np.diag(1/np.sqrt(fv))
#   L2 = np.diag(1/np.sqrt(fv-0.001)) @ D1 @ np.diag(ev+0.001) @ D1.T @ np.diag(1/np.sqrt(fv-0.001))

#   L[0,0] - L2[0,0]

#   (max(fv[[0,1]]) + max(fv[[0,2]]) + max(fv[[0,4]]))*(1/fv[0])

#   max_entry_diff = np.max(abs(L.todense() - L2.todense()))
#   print(f"Max entry diff: {max_entry_diff:.5f}")

#   max_diag_diff = max(abs(L.diagonal() - L2.diagonal()))
#   print(f"Max diagonal diff: {max_diag_diff:.5f}")

#   delta = 2*eps
#   max_deg = max([sum([len(c) == len(f)+1 for c in K.cofaces(f)]) for f in list(faces(K,1))])

#   9*delta

#   max_diff = max(abs(np.linalg.eigvalsh(L.todense()) - np.linalg.eigvalsh(L2.todense())))
#   max_diff <= max_deg*delta**2


  

# def test_matvec_gradient():
#   from pbsig.interpolate import interpolate_family
#   S = simplicial_complex([[0,1,2],[2,3,4]])
#   x_dom = np.linspace(0, 1, 15)
#   F = [lower_star_weight(np.random.uniform(low=0, high=1, size=card(S,0))) for i in range(len(x_dom))]
#   filter_family = [[f(s) for s in S] for f in F]
#   # interpolate_family(filter_family)
#   from scipy.interpolate import CubicSpline, UnivariateSpline
#   from pbsig.betti import weighted_degree

#   fs = { s : CubicSpline(x_dom, [f[i] for f in filter_family]) for i, s in enumerate(S)}
#   fs[Simplex([0,1])].derivative
  
#   D1 = boundary_matrix(S, p=1)
#   L = D1 @ D1.T
#   w0 = np.array([F[0](v) for v in faces(S, 0)])
#   w1 = np.array([F[0](v) for v in faces(S, 1)])
#   WL = np.diag(np.sqrt(w0)) @ D1 @ np.diag(w1) @ D1.T @ np.diag(np.sqrt(w0))
  
#   ## check the weighted degree computation holds for W^{1/2} D1 W D1.T W^{1/2}
#   deg_v = weighted_degree(S, 0, weights=filter_family[0])
#   weight_v = np.array([F[0](v) for v in faces(S, 0)])
#   weight_e = np.array([F[0](e) for e in faces(S, 1)])
#   assert np.allclose(weight_v*deg_v, WL.diagonal())

#   from itertools import product
#   vertices = list(faces(S,0))
#   edges = list(faces(S,1))
#   for v0, v1 in product(faces(S,0), faces(S,0)):
#     if v0 == v1:
#       i = vertices.index(v0)
#       assert np.isclose(deg_v[i]*weight_v[i], WL[i,i])
#     else:
#       i,j = vertices.index(v0), vertices.index(v1)
#       if [v0,v1] in S:
#         k = edges.index(Simplex([v0,v1]))
#         assert np.isclose(WL[i,j],  np.sign(WL[i,j])*np.sqrt(weight_v[i])*weight_e[k]*np.sqrt(weight_v[j]))
#       else:
#         assert np.allclose(WL[i,j], 0)

#   ## Check gradient 
#   alpha = 0.50
#   def param_laplacian(alpha):
#     w0 = np.array([fs[s](alpha).item() for s in faces(S, 0)])
#     w1 = np.array([fs[s](alpha).item() for s in faces(S, 1)])
#     WL = np.diag(np.sqrt(w0)) @ D1 @ np.diag(w1) @ D1.T @ np.diag(np.sqrt(w0))
#     return WL 
#   import numdifftools as nd
  
#   ## Jacobian
#   J = nd.Derivative(param_laplacian, full_output=True)(0.50)
#   i = 0
#   w = np.array([fs[s](0.50) for s in S])
#   w0 = np.array([fs[s](0.50) for s in faces(S,0)])
#   w1 = np.array([fs[s](0.50) for s in faces(S,0)])
#   deg_v = weighted_degree(S, 0, weights=w)

#   ## gradients for the diagonal 
#   alpha = 0.50
#   weighted_degree(S, 0, weights=np.array([fs[s](alpha) for s in S]))
#   grad_vertices = []
#   for i,v in enumerate(faces(S, 0)):
#     sigma = list(faces(list(S.cofaces(v)), 1))
#     dv = sum([fs[s].derivative()(alpha) for s in sigma])*fs[v](alpha) + deg_v[i]*fs[v].derivative()(alpha)
#     grad_vertices.append(dv)

#   print(grad_vertices)
#   print(J.diagonal())

#   from findiff import FinDiff 
#   d_dx = FinDiff(0, 0.0001)
#   d_dx(np.array([param_laplacian(a)[0,0] for a in np.linspace(0.50-1e-8, 0.50+1e-8, 10)]))

#   fs[Simplex([0])].derivative()(0.50)
#   # up_laplacian(S, p=0, weight=F[0], normed=False).todense()

  
#   ## matvec 
#   x = np.random.uniform(size=L.shape[0], low=-1, high=1)
#   L @ x

#   up_laplacian(S, p=0, weight=F[0], normed=True).todense()
  
#   pass

# def test_can_import():
#   from pbsig.linalg import laplacian
#   assert str(type(laplacian)) == "<class 'module'>"

# def test_weighted_boundary_p1():
#   X, S = generate_dataset()
#   D1 = boundary_matrix(S, p=1)

#   ## Ensure nice CSC matrix
#   D1 = D1.tocsc()
#   assert D1.has_sorted_indices, "Must be nice csc matrix"

#   ## Generate weights 
#   wp, wq = [np.random.uniform(size=d, low=0.0, high=1.0) for d in S.shape[:2]]
#   assert all(wp > 0) and all(wq > 0)

#   ## Make boundary matrix weighted
#   W1 = D1.copy()
#   vw = wp[W1.indices]
#   ew = np.repeat(wq, 2)
#   W1.data = np.sign(W1.data)*ew*vw

#   ## Check equivalence of boundary and weighted boundary matrix-vector product 
#   y = np.random.uniform(size=D1.shape[1])
#   assert np.allclose(W1 @ y, (diags(wp) @ D1 @ diags(wq)) @ y)

#   ## Check equivalence of boundary and laplacian matrix-vector product 
#   x = np.random.uniform(size=D1.shape[0])
#   assert np.allclose(W1 @ W1.T @ x, diags(wp) @ D1 @ diags(wq**2) @ D1.T @ diags(wp) @ x), "Matvec failed"
#   assert is_symmetric(W1 @ W1.T)

#   ## Check sqrt of eigen-spectrum of laplacian yields singular values of weighted boundary 
#   LW = W1 @ W1.T
#   ew = np.sort(np.sqrt(np.linalg.eigvalsh(LW.todense())))
#   sw = np.sort(np.linalg.svd(W1.todense())[1])
#   assert np.allclose(ew[1:], sw[1:])

#   ## Check matrix-vector product sqrt
#   W1 = D1.copy()
#   vw = wp[W1.indices]
#   ew = np.repeat(wq, 2)
#   W1.data = np.sign(W1.data)*np.sqrt(ew)*np.sqrt(vw)
#   assert np.allclose(W1 @ W1.T @ x, diags(np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(np.sqrt(wp)) @ x), "Matvec failed"
#   assert is_symmetric(W1 @ W1.T)

#   ## Test symmetry
#   assert is_symmetric(diags(np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(np.sqrt(wp)))
#   assert is_symmetric(diags(1/np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(wp)))
  
# def test_uplaplacian0():
#   X, S = generate_dataset()
#   D1 = boundary_matrix(S, p=1).tocsc()
#   wq = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
#   wpl = np.random.uniform(size=S.shape[0], low=0.0, high=1.0)
#   wpr = np.random.uniform(size=S.shape[0], low=0.0, high=1.0)

#   ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
#   x = np.random.uniform(size=D1.shape[0])
#   LW = diags(wpl) @ D1 @ diags(wq) @ D1.T @ diags(wpr)

#   ## Check diagonal first
#   V, E = list(S.faces(0)), list(S.faces(1))
#   d = np.zeros(S.shape[0])
#   for s_ind, s in enumerate(E):
#     for f in s.boundary():
#       ii = V.index(f)
#       d[ii] += wq[s_ind] * wpl[ii] * wpr[ii]
#   assert np.allclose(d, LW.diagonal())

#   ## Then check the full asymmetric product
#   sgn_pattern = D1[:,[0]].data
#   z = d * x # update with x
#   for s_ind, s in enumerate(E):
#     for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), -sgn_pattern):
#       ii, jj = V.index(f1), V.index(f2)
#       z[ii] += x[jj] * wpl[ii] * wpr[jj] * wq[s_ind] * sgn_ij
#       z[jj] += x[ii] * wpl[jj] * wpr[ii] * wq[s_ind] * sgn_ij
#   assert np.allclose(z, LW @ x)


# def test_up_laplacian1():
#   X, S = generate_dataset()
#   D = boundary_matrix(S, p=2).tocsc()
#   wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
#   wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
#   wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

#   ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
#   x = np.random.uniform(size=S.shape[1])
#   LW = diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr)

#   ## Check diagonal first
#   P, Q = list(S.faces(1)), list(S.faces(2))
#   d = np.zeros(S.shape[1])
#   for s_ind, s in enumerate(Q):
#     for f in s.boundary():
#       ii = P.index(f)
#       d[ii] += wpl[ii] * wq[s_ind] * wpr[ii]
#   assert np.allclose(d, LW.diagonal())

#   ## Then check the full asymmetric product
#   sgn_pattern = D[:,[0]].data ## if vertex-induced
#   z = d * x # update with x
#   for s_ind, s in enumerate(Q):
#     for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), -sgn_pattern):
#       ii, jj = P.index(f1), P.index(f2)
#       z[ii] += sgn_ij * x[jj] * wpl[ii] * wq[s_ind] * wpr[jj]
#       z[jj] += sgn_ij * x[ii] * wpl[jj] * wq[s_ind] * wpr[ii]
#   assert np.allclose(z, LW @ x)

# ## Test with arbitrary sign changes
# def test_up_laplacian1_sgn():
#   X, S = generate_dataset()
#   D = boundary_matrix(S, p=2).tocsc()
#   wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
#   wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
#   wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

#   ## Orient arbitrarily
#   for j in range(D.shape[1]):
#     D[:,[j]] = D[:,[j]]*np.random.choice([-1,1])

#   ## Check equivalence of weighted laplacian matrix-vector product and graph-theoretic version
#   x = np.random.uniform(size=S.shape[1])
#   LW = diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr)

#   ## Check diagonal first
#   P, Q = list(S.faces(1)), list(S.faces(2))
#   d = np.zeros(S.shape[1])
#   for s_ind, s in enumerate(Q):
#     for f in s.boundary():
#       ii = P.index(f)
#       d[ii] += wpl[ii] * wq[s_ind] * wpr[ii]
#   assert np.allclose(d, LW.diagonal())

#   ## Then check the full asymmetric product
#   z = d * x # update with x
#   for s_ind, s in enumerate(Q):
#     sgn_pattern = np.sign(D[:,[s_ind]].data)
#     B = list(s.boundary())
#     for (f1, f2) in combinations(s.boundary(), 2):
#       sgn_ij = sgn_pattern[B.index(f1)]*sgn_pattern[B.index(f2)]
#       ii, jj = P.index(f1), P.index(f2)
#       z[ii] += sgn_ij * x[jj] * wpl[ii] * wq[s_ind] * wpr[jj]
#       z[jj] += sgn_ij * x[ii] * wpl[jj] * wq[s_ind] * wpr[ii]
#   assert np.allclose(z, LW @ x)


# def test_up_laplacian1():
#   X, S = generate_dataset()
#   D = boundary_matrix(S, p=2).tocsc()
#   wq = np.random.uniform(size=S.shape[2], low=0.0, high=1.0)
#   wpl = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)
#   wpr = np.random.uniform(size=S.shape[1], low=0.0, high=1.0)

#   from pbsig.linalg import UpLaplacian
#   L = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
  
#   ## Ensure construction
#   from scipy.sparse.linalg import LinearOperator
#   assert isinstance(L, LinearOperator)
  
#   ## Test unweighted product 
#   x = np.random.uniform(size=L.shape[0])
#   assert np.allclose(D @ D.T @ x, L @ x)
  
#   ## Test weighted product with p+1 simplices
#   L.simplex_weights = wq 
#   assert np.allclose(D @ diags(wq) @ D.T @ x, L @ x)

#   ## Test weighted product with p simplices 
#   L.face_right_weights = wpr 
#   assert np.allclose(D @ diags(wq) @ D.T @ diags(wpr) @ x, L @ x)    
#   L.face_left_weights = wpl 
#   assert np.allclose(diags(wpl) @ D @ diags(wq) @ D.T @ diags(wpr) @ x, L @ x)  

#   ## Test matrix equivalence at API level 
#   from pbsig.linalg import up_laplacian
#   LO = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
#   LM = up_laplacian(S, p=1)
#   assert np.allclose(LO @ x, LM @ x)

#   ## Test matrix equivalence at API level 
#   W = dict(zip(S, np.random.uniform(size=len(S), low=0.0)))
#   LM = up_laplacian(S, p=1, weight = lambda s: W[s])
#   LO = UpLaplacian(list(S.faces(2)), list(S.faces(1)))
#   LO.simplex_weights = np.array([W[s] for s in S.faces(2)])
#   LO.face_left_weights = np.array([W[s] for s in S.faces(1)])
#   LO.face_right_weights = np.array([W[s] for s in S.faces(1)])
#   assert np.allclose(LO @ x, LM @ x)

#   ## Test equivalence via weight function
#   LO = up_laplacian(S, p=1, weight = lambda s: W[s], form='lo')
#   assert np.allclose(LO @ x, LM @ x)


# def test_weighted_up_laplacian():
#   """ 
#   This tests the matrix-vector product and quadratic forms of the pth up-Laplacian Lp = Dq @ Wq @ Dq.T @ Wp
#   where q = p+1, Dq is the q-th boundary matrix, and (Wp,Wq) are diagonal matrices with *non-negative* entries.
#   """
#   X, S = generate_dataset()
#   D1, D2 = boundary_matrix(S, p=(1,2))
  
#   ## Generate weights 
#   wp, wq = [np.random.uniform(size=d, low=0.0, high=1.0) for d in S.shape[:2]]

#   ## Start with the graph Laplacian 
#   L = D1 @ diags(wq) @ D1.T
#   L_up_asym = L @ diags(wp)
#   L_up_sym = diags(np.sqrt(wp)) @ L @ diags(np.sqrt(wp))
#   x = np.random.uniform(size=L_up.shape[0])
#   y = L_up @ x

#   ## Test degree pre-computation first
#   p_faces, q_faces = list(S.faces(p=0)), list(S.faces(p=1))
#   z = np.zeros(S.shape[0]) 
#   for q_ind, q in enumerate(q_faces):
#     for p in boundary(q):
#       p_ind = p_faces.index(p)
#       z[p_ind] += wq[q_ind]
#   assert max(abs(L.diagonal() - z)) < 1e-13

#   ## TODO: test inner product structure
#   # x = np.random.uniform(size=ns[1])
#   # y = np.zeros(ns[1])
#   # del_sgn_pattern = [1,-1,1]
#   # lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(del_sgn_pattern,2)])
#   # for t_ind, t in enumerate(K.faces(p=2)):
#   #   for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
#   #     ii, jj = p_faces.index(e1), p_faces.index(e2)
#   #     c = w0[ii] * w0[jj] * w1[t_ind]**2
#   #     y[ii] += x[jj] * c * s_ij
#   #     y[jj] += x[ii] * c * s_ij


# def test_up_laplacian_native():
#   from pbsig.linalg import laplacian
#   X, K = generate_dataset(15)
#   E = np.array(list(K.faces(1)))
#   nv = K.dim()[0]
#   er = rank_combs(E, n=nv, order="lex")
#   assert isinstance(er, np.ndarray) and len(er) == K.dim()[1]

#   L = laplacian.UpLaplacian0_LS(er, nv)
#   L.fv = np.random.uniform(size=nv)
#   x = np.random.uniform(size=nv)
#   L.prepare()
#   L._matvec(x)

#   from pbsig.linalg import up_laplacian
#   fv = np.array(L.fv.copy())
#   L_up = up_laplacian(K, p=0, weight=lambda s: max(fv[s]) if len(s) == 2 else 1.0)
#   L_up @ x


# def test_laplacian0_cpp():
#   X = np.random.uniform(size=(15,2))
#   S = delaunay_complex(X) 
#   r = rank_combs(S.faces(1), n=S.shape[0], order="lex")
#   x = np.random.uniform(size=L.shape[0])

#   from pbsig.linalg import laplacian, up_laplacian
#   L = laplacian.UpLaplacian0D(r, X.shape[0], X.shape[0])
#   L.compute_indexes()
#   L.precompute_degree()
#   LM = up_laplacian(S, p=0)
#   assert np.allclose(L._matvec(x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)

#   LO = up_laplacian(S, p=0, form='lo')
#   assert np.allclose((LO @ x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)


# def test_laplacian1_cpp():
#   from splex.combinatorial import rank_combs
#   X = np.random.uniform(size=(15,2))
#   S = delaunay_complex(X) 
#   qr = rank_combs(S.faces(2), n=S.shape[0], order="lex")

#   from pbsig.linalg import laplacian, up_laplacian
#   L = laplacian.UpLaplacian1F(qr, S.shape[0], S.shape[1])
#   L.compute_indexes()
#   L.precompute_degree()
  
#   ## Ensure the remappers work
#   assert all(np.ravel(L.simplices == np.array(list(S.faces(2)))))
#   assert all(np.ravel(L.faces == np.array(list(S.faces(1)))))

#   ## Ensure index map is valid
#   pr_true = rank_combs(S.faces(1), n=S.shape[0], order="lex")
#   pr_test = np.array(L.pr)
#   assert len(pr_true) == len(pr_test)
#   assert all((pr_true - pr_test) == 0)

#   ## Check matvec product 
#   x = np.random.uniform(size=S.shape[1])
#   LM = up_laplacian(S, p=1)
#   assert np.allclose(L._matvec(x) - (LM @ x), 0.0, atol=10*np.finfo(np.float32).eps)

#   from pbsig.linalg import UpLaplacian0D


# def test_boundary_faces():
#   from pbsig.linalg import laplacian 
#   T = unrank_combs([0,1,2,3,4], n=10, k=3, order="lex")
#   fr_truth = np.array([rank_combs(boundary(t), n=10, order="lex") for t in T])
#   fr_test = laplacian.decompress_faces([0,1,2,3,4], 10, 2)
#   # rank_combs(, n=10, order="lex")

# def test_pmh():
#   assert True
#   # import perfection
#   # import perfection.czech
#   # X = np.random.uniform(size=(150,2))
#   # S = delaunay_complex(X) 
#   # ranks = rank_combs(S.faces(1))
#   # import cmph
#   # cmph.generate_hash(ranks)
#   # params = perfection.hash_parameters(r)

#   # h = perfection.make_hash(r)
#   # np.sort([h(r) for r in ranks])

#   # f1, f2, G = perfect_hash.generate_hash([str(r) for r in ranks])

#   # perfection.czech.hash_parameters(r)


# def test_diagonal():
#   from pbsig.linalg import laplacian, up_laplacian
#   X = np.random.uniform(size=(150,2))
#   S = delaunay_complex(X) 
  
#   ## Test diagonal dominance, degree and index computations
#   for p in [0,1]:
#     x = np.random.uniform(size=card(S, p))
#     fp = np.random.uniform(size=card(S, p), low=0, high=1)
#     fq = np.array([max(fp[s])*1.05 for s in faces(S, p+1)])
    
#     for i in range(10):
#       ind1 = np.random.choice(range(len(fp)), size=10)
#       ind2 = np.random.choice(range(len(fq)), size=10)
#       fp[ind1] = 0
#       fq[ind2] = 0
#       LO = up_laplacian(S, p=p, form='lo')
#       D1 = boundary_matrix(S, p=p+1)
#       LU = D1 @ diags(fq) @ D1.T
#       LO.set_weights(None, fq, None)
#       np.allclose(LO.diagonal(), LU.diagonal())
#       LO.set_weights(fp, fq, fp)
#       LU = diags(fp) @ D1 @ diags(fq) @ D1.T @ diags(fp)
#       np.allclose(LO @ x, LU @ x)

# def test_rips_laplacian():
#   X = np.random.uniform(size=(8,2))
#   S = rips_complex(X, 2)
  
#   K = rips_filtration(X, p=2)
#   fv = np.random.uniform(size=X.shape[0], low=0.0)
#   f = lambda s: max(fv[s])
#   D = boundary_matrix(K, p=1)
#   fq = np.array([max(fv[e]) for e in faces(K,1)])
#   LU = (diags(fv) @ D @ diags(fq) @ D.T @ diags(fv)).tocoo()
#   LM = up_laplacian(K, p=0, form="array", weight=f)
#   assert all(LM.nonzero()[1] == LU.nonzero()[1])

#   ## 
#   x = np.random.uniform(size=card(S,0), low=0)
#   D = boundary_matrix(K, p=1)
#   LM = up_laplacian(K, p=0, form="array")
#   LO = up_laplacian(K, p=0, form="lo")
#   LU = D @ D.T
#   assert np.allclose(LU @ x, LO @ x)
#   assert np.allclose(LO @ x, LM @ x)

#   LP = UpLaplacianPy(list(S.faces(p+1)), list(S.faces(p)))  
#   LP.set_weights()


# def test_laplacian_derivative():
#   from numdifftools import Derivative
#   family = lambda a: a**2
#   S = simplicial_complex([[0,1,2,3]])
#   L = up_laplacian(S, form="array", p=0)

#   np.random.seed(1234)
#   w = np.random.uniform(size=card(S, 1), low=0, high=1.0)
#   D = boundary_matrix(S, p=1)
#   L_family = lambda a: D @ np.diag(family(a)*w) @ D.T
#   L_deriv = Derivative(L_family)
#   L_deriv(0.001)

#   w_deriv = Derivative(lambda a: family(a)*w)
#   deg = np.diag((D @ D.T).todense())
#   np.diag(w_deriv(0.001)) @ D @ D.T










# def test_laplacian_API():
#   from pbsig.linalg import laplacian, up_laplacian
#   X = np.random.uniform(size=(150,2))
#   S = delaunay_complex(X) 
#   LO = up_laplacian(S, p=0, form='lo')

#   ## Test the ranks are stored correctly
#   for p in [0,1]:
#     LO = up_laplacian(S, p=p, form='lo')
#     assert all(np.ravel(LO.simplices == np.array(list(S.faces(p+1)))))
#     assert all(np.ravel(LO.faces == np.array(list(S.faces(p)))))

#   ## Test they are the same 
#   for p in [0,1]:
#     LM = up_laplacian(S, p=p, form='array')
#     LO = up_laplacian(S, p=p, form='lo')
#     x = np.random.uniform(size=card(S, p))
#     assert np.allclose(LM @ x, LO @ x, atol=10*np.finfo(np.float32).eps)

#   ## Test they match the boundary operator 
#   for p in [0,1]:
#     LO = up_laplacian(S, p=p, form='lo')
#     D1 = boundary_matrix(S, p=p+1)
#     LU = D1 @ D1.T
#     x = np.random.uniform(size=card(S, p))
#     assert np.allclose(LU @ x, LO @ x, atol=10*np.finfo(np.float32).eps)

#   ## Test weighted variants 
#   pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
#   for p in [0,1]:
#     x = np.random.uniform(size=card(S, p))
#     fp = np.random.uniform(size=card(S, p), low=0, high=1)
#     fq = np.array([max(fp[s])*1.05 for s in faces(S, p+1)])
#     LM = up_laplacian(S, p=p, form='array', weight=lambda s: max(fp[s])*1.05)
#     LO = up_laplacian(S, p=p, form='lo', weight=lambda s: max(fp[s])*1.05)
#     D = boundary_matrix(S, p+1)
#     LO.set_weights(None, fq, None)
#     LE = D @ diags(fq) @ D.T
#     assert np.allclose(LO.diagonal(), LE.diagonal())
#     assert np.allclose(LO @ x, LE @ x)
#     LO.set_weights(fp, fq, fp)
#     LE = diags(fp) @ D @ diags(fq) @ D.T @ diags(fp)
#     assert np.allclose(LO.diagonal(), LE.diagonal())
#     assert np.allclose(LO @ x, LE @ x)
#     LU = diags(pseudo(np.sqrt(fp))) @ D @ diags(fq) @ D.T @ diags(pseudo(np.sqrt(fp)))
#     LO.set_weights(pseudo(np.sqrt(fp)), fq, pseudo(np.sqrt(fp)))
#     assert np.allclose(LO @ x, LU @ x, atol=10*np.finfo(np.float32).eps)

#   ## Test normalized operators
#   from scipy.sparse.linalg import eigsh
#   largest_ew = lambda M: eigsh(M, k=1, which='LM', return_eigenvectors=False).item()
  
#   ## Largest eigenvalues should be in [0, p+2]
#   LO = up_laplacian(S, p=0, form='lo', normed=True)
#   assert largest_ew(LO) <= 2.0

#   ## Largest eigenvalues should be in [0, p+2]
#   LO = up_laplacian(S, p=1, form='lo', normed=True)
#   assert largest_ew(LO) <= 3.0

#   ## Weighted version 
#   vertex_weights = np.random.uniform(size=S.shape[0], low=0.0, high=5.0)
#   scalar_product = lambda s: max(vertex_weights[s])
#   for p in [0,1]:
#     LO = up_laplacian(S, p=p, form='lo', weight=scalar_product, normed=False)
#     assert isinstance(LO, LinearOperator)
#     ew_upper = (p+2)*max(LO.diagonal())/min([scalar_product(s) for s in S.faces(p) if scalar_product(s) > 0])
#     assert largest_ew(LO) <= ew_upper
#     LO = up_laplacian(S, p=p, form='lo', weight=scalar_product, normed=True)
#     assert largest_ew(LO) <= p+2.0
  
#   ## Test equivalence of weighting array or operator
#   LM = up_laplacian(S, p=1, form='array', weight=scalar_product)
#   LO = up_laplacian(S, p=1, form='lo', weight=scalar_product)
#   x = np.random.uniform(size=S.shape[1])
#   assert np.allclose(LM @ x, LO @ x, atol=10*np.finfo(np.float32).eps)


#   ## Niceeee bound on maximum eigenvalue: 
#   # (p+2)*np.dot(LM.diagonal(), x**2), 
#   # x = np.random.uniform(size=LM.shape[0])
#   # max(np.linalg.eigvalsh(LM.todense())) <= 3*np.dot(x**2, LM.diagonal()) / np.dot(x**2, np.repeat(1.0, S.shape[1]))

#   # LO.L._matvec(x)


# def test_performance():
#   from pbsig.linalg import laplacian, up_laplacian
#   X = np.random.uniform(size=(500,2))
#   S = delaunay_complex(X) 
  
#   ## Test they are the same 
#   LM = up_laplacian(S, p=0, form='array')
#   LO = up_laplacian(S, p=0, form='lo')
#   x = np.random.uniform(size=S.shape[0])
#   assert np.allclose(LM @ x, LO @ x, atol=10*np.finfo(np.float32).eps)
  
#   import timeit
#   LM = LM.tocsc()
#   timeit.timeit(lambda: LM @ x, number=10000)
#   timeit.timeit(lambda: LO @ x, number=10000)

#   M = np.random.uniform(size=(S.shape[0], S.shape[0]))
#   assert np.allclose(LM @ M, LO @ M, atol=10*np.finfo(np.float32).eps)
  
#   import timeit
#   timeit.timeit(lambda: LM @ M, number=1000)
#   timeit.timeit(lambda: LO @ M, number=100) # about an order of magnitude slower

#   size_LM = LM.data.size * LM.data.itemsize +  LM.indices.size * LM.indices.itemsize +  LM.indptr.size * LM.indptr.itemsize
#   size_LO = len(LO.fq) * 8
#   print(f"size (bytes) of LM: {size_LM} and LO: {size_LO}")

# ## TODO: Verify normalized symmetric laplacian w/ weight function is psd 
# ## and similar per the formula 

# # def test_weighted_linear_L0():
# #   ## Generate random geometric complex
# #   X = np.random.uniform(size=(15,2))
# #   K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
# #   D1, D2 = boundary_matrix(K, p=(1,2))

# #   ## Test weighted linear form (only takes one O(n) vector!)
# #   ns = K.dim()
# #   w0, w1 = np.random.uniform(size=ns[0]), np.random.uniform(size=ns[1])
# #   v0 = np.zeros(ns[0]) # note the O(n) memory
# #   for cc, (i,j) in enumerate(K.faces(1)):
# #     v0[i] += w0[i]**2 * w1[cc]**2
# #     v0[j] += w0[j]**2 * w1[cc]**2
# #   v0 *= x # O(n)
# #   for cc, (i,j) in enumerate(K.faces(1)):
# #     c = w0[i]*w0[j]*w1[cc]**2
# #     v0[i] -= x[j]*c
# #     v0[j] -= x[i]*c
# #   v1 = (diags(w0) @ D1 @ diags(w1**2) @ D1.T @ diags(w0)) @ x
# #   assert np.allclose(v0, v1), "Unweighted Linear form failed!"

# #   ## Test interface method 
# #   from pbsig.linalg import up_laplacian
# #   fv = w0
# #   L_up = up_laplacian(K, p=0, weight=lambda s: max(fv[s]), form='lo')
# #   L_up.simplex_weights
# #   L_up._ws
# #   assert np.allclose(L_up @ x, v0)
# #   assert np.allclose(L_up @ x, v1)


# def test_laplacian_eigsh():
#   X, S = generate_dataset()
#   LM = up_laplacian(S, p=0, form='array')
#   solver = parameterize_solver(LM, solver='jd')
#   ew_m = solver(LM)

#   LO = up_laplacian(S, p=0, form='lo')
#   solver = parameterize_solver(LO, solver='jd')
#   ew_o = solver(LO)

#   ## Ensure eigenvalues are all the same
#   assert np.allclose(ew_o, ew_m, atol=1e-8)
  
# def test_normalized_laplacian():
#   X, S = generate_dataset()
#   D1, D2 = boundary_matrix(S, p=(1,2))
  
#   ## Generate weights 
#   wp = np.random.uniform(size=S.shape[0], low=5.0, high=10.0)
#   wq = np.array([max(wp[i], wp[j]) for i,j in S.faces(1)])

#   ## Start with the graph Laplacian 
#   L = D1 @ diags(wq) @ D1.T
#   assert is_symmetric(L)
#   assert all(np.linalg.eigvalsh(L.todense()) >= -1e-8) # psd check 

#   ## Calculate weighted degree for each vertex 
#   deg = np.zeros(S.shape[0])
#   for cc,(i,j) in enumerate(S.faces(1)):
#     deg[i] += wq[cc]
#     deg[j] += wq[cc]

#   ## Check spectral bound: note these may be negative! (3.5)
#   ew_asym = np.linalg.eigvalsh((diags(1/deg) @ L).todense()) 
#   assert max(ew_asym) <= 0+2 # Should be true, but some negative

#   ## Check Rayleigh quotient (3.6)
#   x = np.random.uniform(size=L.shape[0])
#   WNL = diags(1/deg) @ L  
#   ew_asym = np.linalg.eigvalsh(WNL.todense())
#   assert max(ew_asym) <= 2*(np.dot(x**2, deg) / np.dot(x**2, wp))  

#   ## Check spectrum is bounded in [0,p+2]
#   assert max(np.linalg.eigvalsh(WNL.todense())) <= 2

#   ## Check normalized symmetric is psd (Remark 3.2 in persistent Laplacian)
#   ew_sym = np.linalg.eigvalsh(diags(1/np.sqrt(deg)) @ L @ diags(1/np.sqrt(deg)).todense())
#   assert all(ew_sym >= -1e-8)

#   ## Check we really do have an inner PSD 
#   ew_form1 = (diags(np.sqrt(deg)) @ diags(ew_sym) @ diags(1/np.sqrt(deg))).diagonal()
#   ew_form2 = (diags(1/np.sqrt(deg)) @ diags(ew_sym) @ diags(np.sqrt(deg))).diagonal()
#   assert np.allclose(ew_form1, ew_sym)
#   assert np.allclose(ew_form2, ew_sym)
  
#   ## Check we can get degree by setting vertex weights == 1 
#   LM = up_laplacian(S, weight=lambda s: 1 if len(s) == 1 else max(wp[s]))
#   assert np.allclose(LM.diagonal(), deg)

#   ## Test positive semi definite-ness 
#   eigh = lambda X: np.linalg.eigvalsh(X.todense())
#   is_psd = lambda X: is_symmetric(X) and all(eigh(X) >= -1e-8) 
#   assert is_psd(D1 @ diags(wq) @ D1.T)
#   assert not(is_psd(diags(1/wp) @ D1 @ diags(wq) @ D1.T))
#   assert is_psd(diags(1/np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(wp)))
  
#   ## Test cancellations from left and right
#   s_inner = diags(eigh(diags(1/np.sqrt(wp)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(wp))))
#   assert np.allclose(s_inner.diagonal(), 1/np.sqrt(wp) * s_inner.diagonal() * np.sqrt(wp))

#   ## Normalized version 
#   NWL = diags(1/np.sqrt(deg)) @ D1 @ diags(wq) @ D1.T @ diags(1/np.sqrt(deg))
#   ew_norm = eigh(NWL)
#   assert is_psd(NWL) and max(ew_norm) <= 2

#   ## Try pseudo-inverse by setting vertex weights to 0 
#   deg0 = deg.copy()
#   idx = np.random.choice(np.arange(len(deg0)), size=5, replace=False)
#   deg0[idx] = 0.0
#   pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse

#   ## "Turns off" certain eigenvalues by zero'ing them  
#   ew_full = eigh(D1 @ diags(wq) @ D1.T )
#   ew_zero = pseudo(np.sqrt(deg0)) * ew_full * np.sqrt(deg0)
#   assert np.allclose(ew_full[~np.isclose(ew_zero, 0)], ew_zero[~np.isclose(ew_zero, 0)])

#   ## Normalized
#   eigh(diags(pseudo(deg)) @ D1 @ diags(wq) @ D1.T)
#   eigh(diags(pseudo(np.sqrt(deg))) @ D1 @ diags(wq) @ D1.T @ diags(pseudo(np.sqrt(deg))))

#   ## Not the same as above eigenvalues, but does follow zero-out principle
#   ## And is normalized ! 
#   ew0 = eigh(diags(pseudo(np.sqrt(deg0))) @ D1 @ diags(wq) @ D1.T @ diags(pseudo(np.sqrt(deg0))))
#   ew0_zero = pseudo(np.sqrt(deg0)) * ew0 * np.sqrt(deg0)
#   assert np.allclose(ew0[~np.isclose(ew0_zero, 0)], ew0_zero[~np.isclose(ew0_zero, 0)])
#   assert max(ew0_zero) <= 2.0

#   # edge_idx = np.random.choice(np.arange(len(wq)), size=5, replace=False)
#   # wq_zero = wq.copy()
#   # wq_zero[edge_idx] = 0.0

#   # ## just different eiegenvalues 
#   # eigh(diags(pseudo(np.sqrt(deg0))) @ D1 @ diags(wq) @ D1.T @ diags(pseudo(np.sqrt(deg0))))  
#   # eigh(diags(pseudo(np.sqrt(deg0))) @ D1 @ diags(wq_zero) @ D1.T @ diags(pseudo(np.sqrt(deg0))))

#   ## Inner product matrix result from Lemma 3.3. 
#   from scipy.linalg import solve
#   x, y = np.random.uniform(size=len(wp)), np.random.uniform(size=len(wq))
#   F = diags(wq) @ D1.T @ diags(1/wp)
#   G = diags(1/wq) @ F @ diags(wp) 
#   print(x.T @ F.T @ diags(1/wq) @ y)
#   print(x.T @ diags(1/wp) @ G.T @ y)

#   ## Now, suppose you replace select weights with 0. Do inner products still hold?
#   wp0, wq0 = wp.copy(), wq.copy()
#   wp0[np.random.choice(len(wp0), size=4, replace=False)] = 0
#   wq0[np.random.choice(len(wq0), size=6, replace=False)] = 0
#   F = diags(wq0) @ D1.T @ diags(pseudo(wp0))
#   G = diags(pseudo(wq0)) @ F @ diags(wp0) 
#   print(x.T @ F.T @ diags(pseudo(wq0)) @ y) # cancels wq, includes wp 
#   print(x.T @ diags(pseudo(wp0)) @ G.T @ y) # cancels wp, includes wq
#   ## They do! 

#   ## Conclusion: 
#   # eigh(diags(pseudo(np.sqrt(deg0))) * D1 @ diags(wq) @ D1.T * diags(np.sqrt(deg0)))

#   # L_up_asym = L @ diags(wp)
#   # L_up_sym = diags(np.sqrt(wp)) @ L @ diags(np.sqrt(wp))

#   # B1, B2 = boundary_matrix(S, p = (1, 2))
#   # LG = B1 @ B1.T
#   # ew = np.linalg.eigh(LG.todense())[0]
#   # deg = LG.diagonal()
#   # D = diags(1/np.sqrt(deg))
#   # ew_norm = np.linalg.eigh((D @ LG @ D).todense())[0] 
#   # assert max(ew_norm) <= 2.0
  

# # def test_rank_eq():
  

