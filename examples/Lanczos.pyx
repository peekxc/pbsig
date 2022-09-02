# from scipy.sparse.linalg import LinearOperator, eigsh
# from numpy.typing import ArrayLike
from math cimport floor, sqrt 

cdef rank_C2(int i, int j, int n):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(int x, int n):
  i = int(n - 2 - floor(sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

# nv: int, ne: int
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef D1_matvec_lower_star_lex(float[:] r, float[:] rz, const int[:] E, float[:] fv, float[:] x):
  #r, rz = np.zeros(m), np.zeros(n)
  cdef Py_ssize_t nv = r.shape[0]
  cdef Py_ssize_t ne = rz.shape[0]
  r[:] = 0
  rz[:] = 0
  cdef float cf = 0
  for cc in range(ne):
    i, j = unrank_C2(E[cc], nv)
    cf = fv[j] if fv[i] <= fv[j] else fv[i]
    r[cc] = cf*x[i] - cf*x[j]
  # for cc, (i,j) in enumerate(E):
  #   r[cc] = fe[cc]*x[i] + -fe[cc]*x[j]
  for cc in range(ne):
    i, j = unrank_C2(E[cc], nv)
    rz[i] += fe[cc]*r[cc]
    rz[j] -= fe[cc]*r[cc]
  # for cc, (i,j) in enumerate(E):
  #   rz[i] += fe[cc]*r[cc]
  #   rz[j] -= fe[cc]*r[cc]