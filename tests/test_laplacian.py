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

  ## Check matrix-vector product 
  x = np.random.uniform(size=D1.shape[0])
  assert np.allclose(W1 @ W1.T @ x, diags(wp) @ D1 @ diags(wq**2) @ D1.T @ diags(wp) @ x), "Matvec failed"
  assert is_symmetric(W1 @ W1.T)

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
  
  LG = D1 @ diags(wq) @ D1.T 
  L_inner = diags(1/np.sqrt(wp)) @ LG @ diags(1/np.sqrt(wp))
  L_full = diags(np.sqrt(wp)) @ L_inner @ diags(1/np.sqrt(wp))
  assert is_symmetric(L_inner)
  assert not(is_symmetric(L_full))
  assert np.allclose(abs((LG @ diags(1/wp)) - L_full).data, 0.0)

  L_inner = diags(np.sqrt(wp)) @ LG @ diags(np.sqrt(wp))
  L_full = diags(1/np.sqrt(wp)) @ L_inner @ diags(np.sqrt(wp))
  assert is_symmetric(L_inner)
  assert not(is_symmetric(L_full))
  assert np.allclose(abs((LG @ diags(wp)) - L_full).data, 0.0)


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


from pbsig.linalg import _up_laplacian_matvec_1
E = np.array(list(K.faces(1)))
w0 = fv 
w1 = fv[E].max(axis=1)
f = _up_laplacian_matvec_1(list(K.faces(1)), w0, w1)
f(x)

up_laplacian(K, )

## Generate random geometric complex
X = np.random.uniform(size=(15,2))
K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
D1, D2 = boundary_matrix(K, p=(1,2))

## Test unweighted/combinatorial quadratic form
x = np.random.uniform(low=-0.50, high=0.50, size=D1.shape[0])
v0 = sum([(x[i] - x[j])**2 for i,j in K.faces(1)])
v1 = x @ (D1 @ D1.T) @ x
assert np.isclose(v0, v1), "Unweighted Quadratic form failed!"

## Test unweighted/combinatorial linear form
ns = K.dim()
w0,w1 = np.ones(ns[0]), np.ones(ns[1])
r, v0 = np.zeros(ns[1]), np.zeros(ns[0])
for cc, (i,j) in enumerate(K.faces(1)):
  r[cc] = w1[cc]*w0[i]*x[i] - w1[cc]*w0[j]*x[j]
for cc, (i,j) in enumerate(K.faces(1)):
  v0[i] += w1[cc]*r[cc] #?? 
  v0[j] -= w1[cc]*r[cc]
v1 = (D1 @ D1.T) @ x
assert np.allclose(v0, v1), "Unweighted Linear form failed!"

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


def test_up_laplacian_cpp():
  X = np.random.uniform(size=(15,2))
  K = delaunay_complex(X) # assert isinstance(K, SimplicialComplex)
  D1, D2 = boundary_matrix(K, p=(1,2))
  
  ns = K.dim()
  w0,w1 = np.random.uniform(size=ns[1], low=0.0), np.random.uniform(size=ns[2], low=0.0)

  from pbsig.linalg import laplacian
  laplacian.UpLaplacian0_LS

# def up_laplacian(K: Union[SparseMatrix, SimplicialComplex], p: int = 0, normed=False, return_diag=False, form='array', dtype=None, **kwargs):
## Up Laplacian - test weighted linear form (only takes one O(n) vector!) 
ns = K.dim()
w0,w1 = np.random.uniform(size=ns[1]), np.random.uniform(size=ns[2])

P = np.sign((D2 @ D2.T).tocsc().data)
LW = (diags(w0) @ D2 @ diags(w1**2) @ D2.T @ diags(w0)).tocsc()
assert all(np.sign(L2.data) == P)

p_faces = list(K.faces(p=1))
z = np.zeros(ns[1]) 
for t_ind, t in enumerate(K.faces(p=2)):
  for e in t.boundary():
    e_ind = p_faces.index(e)
    z[e_ind] += w1[t_ind]**2 * w0[e_ind]**2
assert max(abs(LW.diagonal() - z)) < 1e-13

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

#assert max(abs((abs(L2) @ x) - (y + z*x))) <= 1e-13, "Failed signless comparison"
assert max(abs((L2 @ x) - (y + x*z))) <= 1e-13, "Failed matrix vector"

## Using only one O(n) vector!
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
for t_ind, t in enumerate(K.faces(p=2)):
  for e in t.boundary():
    ii = p_faces.index(e)
    y[ii] += x[ii] * w1[t_ind]**2 * w0[ii]**2
assert max(abs((LW @ x) - y)) <= 1e-13, "Failed matrix vector"

## Test the up-laplacian matrices and matrix free methods yield the same results 
from pbsig.linalg import up_laplacian
L1 = up_laplacian(K, w0=w0, w1=w1**2, p=1)
assert max(abs((L1 @ x) - y)) <= 1e-13

L1 = up_laplacian(K, w0=w0, w1=w1**2, p=1, form='lo')
assert max(abs((L1 @ x) - y)) <= 1e-13



## Apply a directional trasnform 
from pbsig.linalg import eigsh_family
from pbsig.pht import rotate_S1, uniform_S1

V, E = np.array(list(K.faces(0))).flatten(), np.array(list(K.faces(1)))
fv, fe = lambda v: (X @ v)[V] + 1.0, lambda v: (X @ v)[E].max(axis=1) + 1.0
L0_dt = (up_laplacian(K, fv(v), fe(v), p=0) for v in uniform_S1(32))
R = list(eigsh_family(L0_dt, p = 0.50))


## Make a directional transform class?

## Weights should be non-negative to ensure positive semi definite?
#np.linalg.eigvalsh(up_laplacian(K, p=0).todense())
#np.linalg.eigvalsh((diags(w0) @ up_laplacian(K, p=0) @ up_laplacian(K, p=0)).todense())


list(eigsh_family(L0_dt, p = 1.0, reduce=sum))

LO = next(L0_dt)
trace_threshold(LO, 1.0)

LO = up_laplacian(K, w0, w1, p=0, form='lo')

# I = np.zeros(LO.shape[0])
# d = np.zeros(LO.shape[0])
# for i in range(LO.shape[0]):
#   I[i] = 1
#   I[i-1] = 0
#   d[i] = (LO @ I)[i]

# I = I.tocsc()



[(I[:,[i]].T @ LO @ I[:,[i]]).data[0] for i in range(LO.shape[0])]

I[:,[i]].todense()

G = {
  'vertices' : np.ravel(np.array(list(K.faces(0)))),
  'edges' : np.array(list(K.faces(1))),
  'triangles' : np.array(list(K.faces(2)))
}

y + z*x

# np.sign(L2.tocsc().data)

v1 = L2 @ x


# e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv)) ## Can be cast using minimal perfect hashing function
# deg_e *= (fe**2)
assert np.allclose(deg_e - LT_ET.diagonal(), 0.0), "Collecting diagonal terms failed!"

y = np.zeros(len(x))
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv))
  for (ii,jj),s_ij in zip(combinations(e_ind, 2), [-1,1,-1]):
    ## The sign pattern is deterministic! 
    v = (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)
    y[ii] += x[jj] * v * s_ij
    y[jj] += x[ii] * v * s_ij






## Test linear form
L = graph_laplacian(K)
y = L.diagonal() * x # np.zeros(shape=len(x))
for i,j in K['edges']:
  y[i] -= x[j]
  y[j] -= x[i]
assert all(np.isclose(L @ x, y)), "Linear form failed!"

## Test edge-weighted version: LE = \hat{\partial_1}\hat{\partial_1}^T = \partial_1 W^2 \partial_1^T
fe = np.random.uniform(size=K['edges'].shape[0], low=0.0, high=1.0)
W = diags(fe)
D1_E = D1.copy().tocsc()
D1_E.data = np.sign(D1_E.data) * np.repeat(fe, 2)
LE = D1 @ W**2 @ D1.T
assert np.allclose((LE - (D1_E @ D1_E.T)).data, 0), "edge-weighed Laplacian form wrong!"

## Test edge-weighted matrix-vector product
d = LE.diagonal() # the row sums of LE not including diagonal
y = d * x
for cc, (i,j) in enumerate(K['edges']):
  y[i] -= fe[cc]**2 * x[j]
  y[j] -= fe[cc]**2 * x[i]
assert np.allclose(y, LE @ x), "edge-weighted Laplacian linear form wrong!"

## Test L_2^up entry-wise
nv = len(G['vertices'])
E, T = G['edges'], G['triangles']
S = (D2 @ D2.T).copy() ## Precompute sign matrix
S.data = np.sign(S.data)
fe = np.random.uniform(size=E.shape[0], low=0.0, high=1.0)
ft = np.random.uniform(size=T.shape[0], low=0.0, high=1.0)
D2_ET = D2.copy().tocsc()
D2_ET.data = np.sign(D2_ET.data) * ft[np.repeat(range(D2.shape[1]), 3)]*fe[D2_ET.indices]
LT_ET = D2_ET @ D2_ET.T 

## (a) Test entry-wise from weighted boundary matrix is (+/-1)*fi*fj*f(cofacet(i,j))**2
from pbsig.utility import rank_combs, rank_comb
cofacet = lambda i,j: np.sort(np.unique(np.append(E[i,:], E[j,:])))
tr = rank_combs(T, n=len(K['vertices']), k=3)
for i,j in edge_iterator(LT_ET):
  sigma_ind = np.searchsorted(tr, rank_comb(cofacet(i,j), n=nv, k=3)) # this could be done in O(1) time via minimal perfect hashing
  assert abs(LT_ET[i,j]-S[i,j]*fe[i]*fe[j]*ft[sigma_ind]**2) <= np.sqrt(1e-15)

## (b) Test the diagonal entries match fii**2 * (sum(f(cofacets(eii))**2))
cofacets = lambda i,j: T[np.logical_and(T[:,0] == i, T[:,1] == j) | np.logical_and(T[:,1] == i, T[:,2] == j) | np.logical_and(T[:,0] == i, T[:,2] == j),:] 
for cc, (i,j) in enumerate(E):
  sigma_ind = np.searchsorted(tr, rank_combs(cofacets(i,j), n=nv, k=3))
  #print(sum([(ft[s]*fe[cc])**2 for s in sigma_ind]))
  assert abs(LT_ET[cc,cc] - (fe[cc]**2)*sum(ft[sigma_ind]**2)) <= np.sqrt(1e-15)
  
## (c) Now test the matrix form LT_ET == We * D2 * Wt^2 * D2 * We
Wt, We = diags(ft), diags(fe)
assert np.allclose((LT_ET - (We @ D2 @ Wt**2 @ D2.T @ We)).data, 0)

## (d) Formulate the naive matrix-vector product that enumerates edges 

## (e) Formulate the O(m) matrix-vector product that enumerates triangles
from pbsig.utility import *
nv = len(G['vertices'])
E, T = G['edges'], G['triangles']
fe,ft = w0,w1
er = rank_combs(E, n=len(G['vertices']), k=2)
x = np.random.uniform(size=E.shape[0], low=0.0, high=1.0)
deg_e = np.zeros(len(x)) 
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv)) ## Can be cast using minimal perfect hashing function
  deg_e[e_ind] += ft[t_ind]**2 * (fe[e_ind]**2) ## added fe

# deg_e *= (fe**2)
assert np.allclose(deg_e - LT_ET.diagonal(), 0.0), "Collecting diagonal terms failed!"

y = np.zeros(len(x))
for t_ind, (i,j,k) in enumerate(T):
  e_ind = np.searchsorted(er, rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv))
  for (ii,jj),s_ij in zip(combinations(e_ind, 2), [-1,1,-1]):
    ## The sign pattern is deterministic! 
    v = (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)
    y[ii] += x[jj] * v * s_ij
    y[jj] += x[ii] * v * s_ij
  # for ii in e_ind:
  #   y[ii] += x[jj] * (fe[ii]) * (fe[jj]) * (ft[t_ind]**2)

    #y[jj] += x[ii] * (fe[jj]**2) * (ft[t_ind]**2)
  # for ii,jj in combinations(e_ind, 2):
  #   y[ii] += x[ii] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2
  #   y[jj] += x[jj] * S[ii,jj]*fe[ii]*fe[jj]*ft[t_ind]**2

assert max(abs((LT_ET @ x) - (y + deg_e*x))) <= np.sqrt(1e-16), "Weighted up-Laplacian matrix-vector product failed"
# (abs(LT_ET.todense()) @ x)[:5]

z = LT_ET @ x

s = rank_combs(K['edges'], n=nv, k=2) % 2

for i,j in combinations(range(nv), 2):
  if L2[i,j] != 0:
    print(f"sign: {np.sign(L2[i,j])}, i: {s[i]}, j: {s[j]}")

from pbsig.utility import parity
def boundary_sgn(face, coface):
  return parity(np.append(face, np.setdiff1d(coface, face)))



nv = len(K['vertices'])
E_ind = np.sort(rank_combs(K['edges'], k=2, n=nv))
z = np.zeros(len(x))
for t in K['triangles']:
  i,j,k = np.sort(t)
  ii,ji,ki = np.searchsorted(E_ind, rank_combs([[i,j],[i,k],[j,k]],k=2,n=nv))
  z[ii]
