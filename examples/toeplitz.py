import numpy as np 

def toeplitz(x):
  n = len(x)
  M = np.zeros(shape=(n, n))
  for i in range(n):
    if i == 0: 
      M += np.eye(n, k=0)*x[0]
    else:
      M += x[i]*np.eye(n, k=i) + x[i]*np.eye(n, k=-i)
  return(M)

# https://mathoverflow.net/questions/238113/finding-toeplitz-matrix-nearest-to-a-given-matrix
def project_toeplitz(A):
  assert A.shape[0] == A.shape[1], "invalid A"
  n = A.shape[0]
  x = np.zeros(n)
  for i in range(n):
    if i == 0: 
      x[i] = np.trace(A.T @ np.eye(n, k=0)) / np.trace(np.eye(n, k=0).T @ np.eye(n, k=0))
    else:
      M = np.eye(n, k=i) + np.eye(n, k=-i)
      x[i] = np.trace(A.T @ M) / np.trace(M.T @ M)
  return(x)


A = np.random.uniform(size=(5,5))
A = A.T @ A

t = project_toeplitz(A)
T = toeplitz(t)
np.linalg.norm(T - A)

import matplotlib.pyplot as plt
plt.scatter(range(len(t)), t)


from scipy.stats import linregress
m = linregress(np.fromiter(range(len(t)), dtype=int), t).slope

