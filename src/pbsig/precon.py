import numpy as np 
from typing import * 
from numpy.typing import ArrayLike
from scipy.sparse import tril, diags
from scipy.sparse.linalg import *
from scipy.sparse import diags 
import networkx as nx
# L = (D1 @ D1.T)
# # G = nx.fast_gnp_random_graph(30, p=0.20)
# G = nx.connected_watts_strogatz_graph(n=150, k=25, p=0.30)
from pbsig.linalg import effective_resistance
from pbsig.simplicial import graph_laplacian, edge_iterator
from networkx import Graph


class ShiftedJacobi(LinearOperator):
  def __init__(self, A, shift_f: Callable):
    self.D = A.diagonal()
    self.dtype, self.shape = A.dtype, A.shape
    self.shift_f = shift_f
  def _matvec(self, x):
    return np.ravel(x.flatten()) * np.reciprocal(self.D)
  def _matmat(self, X):
    shifts = shift_f() # primme.get_eigsh_param('ShiftsForPreconditioner')
    Y = np.copy(X)
    for i in range(X.shape[1]): 
      Y[:,i] = X[:,i] / (self.D - shifts[i])
    return Y 

def Jacobi(A):
  DI = diags(np.reciprocal(A.diagonal()))
  return aslinearoperator(DI)

def ssor(A, omega: float = 1.0):
  L, D = tril(A, k=1), diags(A.diagonal())
  LHS = (1/omega)*D + L
  return aslinearoperator(omega/(2-omega) * (LHS @ diags(np.reciprocal(A.diagonal())) @ LHS.T))
import primme
class Minres(LinearOperator):
  def __init__(self, A, tol=1e-15, shift_f = lambda: 0):
    self.A = A
    self.dtype, self.shape = A.dtype, A.shape
    self.tol = tol
    self.shift_f = shift_f
  def _matvec(self, x):
    try:
      shifts = primme.get_eigsh_param('ShiftsForPreconditioner')[0]
    except RuntimeError:
      shifts = 0
      print("failed to get shifts")
    #shifts = self.shift_f()
    return minres(A=self.A, b=x, tol=self.tol, shift=shifts)[0]
  def _matmat(self, X):
    try:
      shifts = primme.get_eigsh_param('ShiftsForPreconditioner')[0]
    except RuntimeError:
      shifts = np.zeros(X.shape[0])
    #shifts = self.shift_f() # primme.get_eigsh_param('ShiftsForPreconditioner')
    Y = X.copy()
    for sigma, i in zip(shifts, range(X.shape[1])): 
      Y[:,i] = minres(A=self.A, b=X[:,i], tol=self.tol, shift=sigma)[0]
    return Y 

def _sparsifier_constant(n: int, eps: float, C: int = 1, c: int = 9) -> int:
  assert 1/np.sqrt(n) <= eps and eps <= 1, "Invalid epsilon choice"
  return int(np.floor(c*C**2*(n*np.log(n))/eps**2))




def graph_sparsifier(L, epsilon: Union[float, str] = ["tightest", "2-approx"], max_edges: int = np.inf, **kwargs):
  """ 
  Given graph Laplacian L = D - A representing a graph G= (V, E), produces a weighed subgraph H = (V, E', w) such that, with probability at least 1/2 and for sufficiently large n:

  (1 - epsilon) x^T L_G x <= x^T L_H x <= (1 + epsilon) x^T L_G x     (for all x)

  where (L_G, L_H) are the unweighted/weighted graph Laplacians of (G,H), respectively. The graph H is called a *spectral sparsifier* of G. 

  Parameters: 
    G := networkx Graph 
    epsilon := an accuracy parameter in the range 1/sqrt(n) < epsilon <= 1, or one of "tightest" or "2-approx" to set a default
    max_edges := limit on the maximum edges in H
  
  Additional keyword arguments are passed to the sparsifier_constant() function. 
  
  Based on the algorithm from: https://arxiv.org/pdf/0803.0929.pdf
  """
  # assert isinstance(G, nx.Graph), "Must be networkx graph"
  # nv, ne = G.number_of_nodes(), G.number_of_edges()
  nv, ne = L.shape[0], int((L.nnz - L.shape[0])/2) 
  
  ## Determine epsilon 
  k = ne
  if isinstance(epsilon, list) and epsilon == ["tightest", "2-approx"]:
    k = _sparsifier_constant(nv, 1.0/np.sqrt(nv))
  elif isinstance(epsilon, str) and epsilon == "tightest":
    k = _sparsifier_constant(nv, 1.0/np.sqrt(nv))
  elif isinstance(epsilon, str) and epsilon == "2-approx":
    k = _sparsifier_constant(nv, 1.0)
  elif isinstance(epsilon, float):
    k = _sparsifier_constant(nv, epsilon, **kwargs)
  else: 
    raise ValueError("Invalid epsilon supplied")
  
  ## Get effective resistance of unweighted graph Laplacian
  er = effective_resistance(L) # add edge weights if G is weighted
  p = er/sum(er) 

  ## Sampling procedure
  max_edges = min(max_edges, ne)
  weights, nnz = np.zeros(ne), 0
  for i in range(k):
    ei = np.random.choice(range(ne), p=p, replace=True)
    nnz += 1 if weights[ei] == 0.0 else 0
    weights[ei] += 1
    if nnz >= max_edges:
      break
  num_sampled = i
  weights *= (1.0/num_sampled*p) # add edge weights if G is weighted
  
  ## Make the sparsifier 
  H = nx.Graph()
  H.add_nodes_from(range(nv))
  H.add_weighted_edges_from([(i, j, w) for (i,j), w in zip(edge_iterator(L), weights) if w > 0])
  return H


# class SSOR(LinearOperator):
#   def __init__(self, A, omega: float = 1.0):
#     self.L = tril(A, k-1)
#     self.D = diags(A.diagonal())
#     self.dtype = A.dtype
#     self.shape = A.shape
#     self.omega = omega
#   def _matvec(self, v):
#     c = self.omega/(2-self.omega)
#     LHS = (1/self.omega)*self.D + self.L
#     return LHS @ (1.0/self.D) @ LHS.T
    
  
class PCG_OPinv(LinearOperator):
  def __init__(self, A, shift_f: Callable):
    self.shape, self.dtype = A.shape, A.dtype
    self.A = A 
    self.I = diags(np.repeat(1.0, self.shape[0]))
    self.n_calls = 0
    self.cg = cg
    self.shift_f = shift_f
  def __update_cb__(self, xk: ArrayLike): 
    self.n_calls += 1
  # def _get_shifts():
  #   self.shifts = primme.get_eigsh_param('ShiftsForPreconditioner')
  def _matvec(self, x):
    if x.ndim == 1: 
      return x / self.A.diagonal()
    y = np.copy(x)
    self._get_shifts()
    for i in range(x.shape[1]):
      # S = shifts[i]*self.I #S = np.array([s if s > 0 else 1 for s in S])
      y[:,i] = self.cg(A=self.M - self.shifts[i]*self.I, b=x, tol=1e-6, callback=self.__update_cb__)[0]
    return y