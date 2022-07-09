
np.sum(np.linalg.svd(D1.A, compute_uv=False))


np.sum(np.linalg.svd(D2.A, compute_uv=False))
np.sum(pca(D2, k=np.min(D1.shape), raw=True, n_iter=15)[1])

from numpy.typing import ArrayLike
# https://gregorygundersen.com/blog/2019/01/17/randomized-svd/
def rsvd(X: ArrayLike, k: int, n_steps: int = 2, p: Optional[int] = None, **kwargs):
  """
  Randomized SVD generated

  k 
  """
  m,n = X.shape
  X = X.T if m < n else X
  O = np.random.normal(0.0, 1.0, size=(X.shape[1], k))
  Y, _ = np.linalg.qr(X @ O)

  ## Randomized subspace iteration  
  for i in range(n_steps):
    Y, _ = np.linalg.qr(X.T @ Y)
    Y, _ = np.linalg.qr(X @ Y)

  B = Y.T @ X
  u, S, Vt = np.linalg.svd(B, full_matrices=False)
  U = Y @ u 
  return((U, S, Vt) if m < n else (Vt.T, S, U.T))


np.linalg.svd(D1.A, compute_uv=False)[:10]
rsvd(D1, k=10, n_steps=12)[1]

pca(D1, k=10, raw=False, n_iter=12)[1]

np.sum(pca(D1, k=np.min(D1.shape), raw=False, n_iter=10)[1])



np.random.uniform(size=())