import numpy as np
from numpy.typing import ArrayLike
from typing import * 
from scipy.spatial.distance import cdist, pdist, squareform


def dist(x: ArrayLike, y: Optional[ArrayLike] = None, paired = False, metric : Union[str, Callable] = 'euclidean', **kwargs):
	"""Provides a common framework to calculate distances.
	
	If x and y are  (n x d) and (m x d) numpy arrays, respectively, then:
	Usage:  
		(1) dist(x)                         => (n choose 2) pairwise distances in x
		(2) dist(x, paired = True)          => (n x n) distance matrix              
		(3) dist(x, y)                      => (n x m) distances between x and y. If x == y, then equivalent to (2).
		(4) dist(x, y, paired = True)       => (n x 1) individual pairwise distances between x and y (requires n == m).
		(5) dist(..., metric = "correlation") => any of (1-4) specialized with a given metric
	The supplied 'metric' can be either a string or a real-valued binary distance function.  
	
  As this function acts as a wrapper around the 'cdist' and 'pdist' functions from the scipy.spatial.distance module, 
  both the metric and the additional keyword arguments are passed to 'pdist' or 'cdist', respectively. 
  """
	x = np.asanyarray(x) ## needs to be numpy array
	if (x.shape[0] == 1 or x.ndim == 1) and y is None: 
		return(np.zeros((0, np.prod(x.shape))))
	if y is None:
		#return(cdist(x, x, metric, **kwargs) if (as_matrix) else pdist(x, metric, **kwargs))
		return(squareform(pdist(x, metric, **kwargs)) if paired else pdist(x, metric, **kwargs))
	else:
		n, y = x.shape[0], np.asanyarray(y)
		if paired:
			if x.shape != y.shape: raise Exception("x and y must have same shape if paired=True.")
			return(np.array([cdist(x[ii:(ii+1),:], y[ii:(ii+1),:], metric, **kwargs).item() for ii in range(n)]))
		else:
			if x.ndim == 1: x = np.reshape(x, (1, len(x)))
			if y.ndim == 1: y = np.reshape(y, (1, len(y)))
			return(cdist(x, y, metric, **kwargs))
	