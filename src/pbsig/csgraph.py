import numpy as np
from scipy.sparse import sparray
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform

def knn(X: np.ndarray, k: int, method: str = 'auto', sort: bool = True, **kwargs):
  if method == 'kdtree' or method == 'auto' and X.shape[1] <= 16: 
    X_kdtree = KDTree(X)
    return X_kdtree.query(X, k=k)
  else: 
    knn_dist = squareform(pdist(X, **kwargs))
    knn_indices = knn_dist.argpartition(kth=k, axis=1)[:,:k] if not sort else knn_dist.argsort(axis=1)[:,:k]
    knn_dist = np.array([knn_dist[cc,ind] for cc, ind in enumerate(knn_indices)])
    return knn_dist, knn_indices

def neighborhood_graph(X: np.ndarray, k: int, weighted: bool = False, **kwargs) -> sparray:
  """Returns the (possibly weighted) adjacency matrix representing the k-nearest neighborhood graph of X."""
  from scipy.sparse import coo_array
  from array import array
  knn_dist, knn_ind = knn(X, k+1)
  nv = X.shape[0]
  I, J = np.repeat(np.arange(len(X)), k), np.ravel(knn_ind[:,1:])
  if weighted: 
    A = coo_array((np.ravel(knn_dist[:,1:]), (I, J)), shape=(nv,nv), dtype=float)
    return A 
  else:
    A = coo_array((np.repeat(True, len(I)), (I, J)), shape=(nv,nv), dtype=bool)
    return A