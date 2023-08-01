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
from itertools import chain
import networkx as nx
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

def test_psd():
  from scipy.sparse import random 
  from pbsig.linalg import PsdSolver
  A = random(150,150, density=0.05)
  A = A.T @ A
  solver = PsdSolver(tol=1e-12, k=14, eigenvectors=True)
  solver(A)
  solver(A.todense())
  solver(A, solver="gd")
  solver.test_accuracy(A)

def test_operators():
  G = nx.Graph()
  while (len(list(nx.connected_components(G))) != 1):
    G = nx.random_geometric_graph(100, radius=0.15, dim=2)
  S = simplicial_complex(chain(G.nodes(), G.edges()), form="tree")
  S.expand(2)
  fv = np.random.uniform(size=card(S,0), low=0, high=0.01)

  ## Normalized Laplacians are quite stable
  LM = up_laplacian(S, p=0, form='array', weight=lower_star_weight(fv), normed=True)
  LO = up_laplacian(S, p=0, form='lo', weight=lower_star_weight(fv), normed=True)
  ew0_ar = eigsh(LM, k=30)[0]
  ew0_lo = eigsh(LO, k=30)[0]
  assert np.allclose(ew0_ar - ew0_lo, 0.0)

  LM = up_laplacian(S, p=1, form='array', weight=lower_star_weight(fv), normed=True)
  LO = up_laplacian(S, p=1, form='lo', weight=lower_star_weight(fv), normed=True)
  ew0_ar = eigsh(LM, k=30)[0]
  ew0_lo = eigsh(LO, k=30)[0]
  assert np.allclose(ew0_ar - ew0_lo, 0.0)

def test_matvec():
  for i in range(30):
    G = nx.Graph()
    while (len(list(nx.connected_components(G))) != 1):
      G = nx.random_geometric_graph(100, radius=0.15, dim=2)
    S = simplicial_complex(chain(G.nodes(), G.edges()))
    LO = up_laplacian(S, p = 0, form='lo')
    LM = up_laplacian(S, p = 0, form='array').tocsr()
    for i in range(1000):
      x = np.random.uniform(low=0, high=1, size=LO.shape[0])
      assert max(abs((LM @ x) - (LO @ x))) <= 1e-14


fe = np.array([lower_star_weight(fv)(e) for e in faces(S,1)])
np.array(LO.set_weights(np.sqrt(fv), fe, np.sqrt(fv)).degrees)[:10]
np.array(LO.set_weights(np.ones(len(fv)), fe, np.ones(len(fv))).degrees)[:10]

noise = np.random.uniform(size=len(fv), low=-1e-8, high=1e-8)
LO_perturb = up_laplacian(S, p=0, form='lo', weight=lower_star_weight(fv + noise), normed=True)
max(abs(eigsh(LO_perturb, k=30)[0] - ew0))

np.where(np.isclose(np.sqrt(LO.degrees), 0.0), 0, 1/np.sqrt(LO.degrees))

up_laplacian(S)

solver = eigvalsh_solver(LO)
solver(LO)
import primme

ew, ev, info = primme.eigsh(LM, k=16, ncv=20, return_eigenvectors=True, return_stats = True, which="LM", method='PRIMME_GD', maxiter=25000)
primme.eigsh(LM, k=15, ncv=20, lock=ev[:,[-1]], return_eigenvectors=True, return_stats = True, which="LM", method='PRIMME_GD', maxiter=25000)

primme.eigsh(LM, k=15, ncv=20, return_eigenvectors=False, return_stats = True, which="LM", method='PRIMME_GD', maxiter=25000)
primme.eigsh(LO, k=15, ncv=20, return_eigenvectors=False, return_stats = True, which="LM",method="PRIMME_GD", maxiter=25000)

q,r = np.linalg.qr(np.c_[np.ones(LO.shape[0])])

# LO.face_left_weights = np.random.uniform(size=100, low=0, high=10)
# LO.face_right_weights = np.array(LO.face_left_weights)

# fv = np.array(LO.face_left_weights)
# LO.simplex_weights = fv[LO.simplices].max(axis=1)

# eigsh(LO, k=15, ncv=20)[0]

for i in range(30):
  G = nx.Graph()
  while (len(list(nx.connected_components(G))) != 1):
    G = nx.random_geometric_graph(100, radius=0.15, dim=2)
  S = simplicial_complex(chain(G.nodes(), G.edges()))
  LO = up_laplacian(S, p = 0, form='lo')
  LM = up_laplacian(S, p = 0, form='array').tocsr()
  for i in range(1000):
    x = np.random.uniform(low=0, high=1, size=LO.shape[0])
    assert max(abs((LM @ x) - (LO @ x))) <= 1e-14

  ## There is a multiplicity issue (but not a precision issue!) 
  print(max(abs(eigh(LM.todense())[0][-15:] - eigsh(LO, k=15, ncv=20)[0])))



