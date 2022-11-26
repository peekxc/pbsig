import numpy as np 
from typing import * 
from numpy.typing import ArrayLike
from scipy.sparse import tril, diags
from scipy.sparse.linalg import cg, LinearOperator, aslinearoperator

class ShiftedJacobi(LinearOperator):
  def __init__(self, A, shift_f: Callable):
    self.D = A.diagonal()
    self.dtype, self.shape = A.dtype, A.shape
    self.shift_f = shift_f
  def _matvec(self, x):
    return np.ravel(x.flatten()) * np.reciprocal(self.D)
  def _matmat(self, X):
    # shifts = shift_f() # primme.get_eigsh_param('ShiftsForPreconditioner')
    Y = np.copy(X)
    for i in range(X.shape[1]): 
      Y[:,i] = X[:,i] / self.D # - shifts[i])
    return Y 

def Jacobi(A):
  DI = diags(np.reciprocal(A.diagonal()))
  return aslinearoperator(DI)

def ssor(A, omega: float = 1.0):
  L, D = tril(A, k=1), diags(A.diagonal())
  LHS = (1/omega)*D + L
  return aslinearoperator(omega/(2-omega) * (LHS @ diags(np.reciprocal(A.diagonal())) @ LHS.T))




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