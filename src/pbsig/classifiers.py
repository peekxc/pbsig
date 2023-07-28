from itertools import *
from typing import * 
from numpy.typing import ArrayLike
import numpy as np
from pbsig.shape import shells
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

def subsample_hash(X: np.ndarray, size: int = 100) -> int:
  """Hashes the byte representation of a numpy array by random-sampling."""
  rng = np.random.RandomState(89)
  inds = rng.randint(low=0, high=X.size, size=size)
  b = X.flat[inds]
  b.flags.writeable = False
  return hash(b.data.tobytes())

class ShellsClassifier(BaseEstimator, ClassifierMixin):
  """ Weak classifier that uses a histogram of distances to perform classification """
  def __init__(self, bins: Union[int, Iterable] = 10, dim: int = 2, random_state = None, cache: bool = False):
    self.random_state = random_state # np.random.RandomState(random_state) # used for sampling shells
    self.bins = bins
    self.dim = dim  # ambient dimension of the point cloud 
    self.cache = cache # whether to use a cache for each pair (X,y) s.t. fitting (X,y,w) is faster
  
  def shells_signature(self, x: ArrayLike, bins: Optional[ArrayLike] = None) -> ArrayLike:
    bins = self.bins_ if bins is None else bins
    x_pc = x.reshape((len(x) // self.dim, self.dim))
    barycenter = x_pc.mean(axis=0)
    dist_to_barycenter = np.linalg.norm(x_pc - barycenter, axis=1)
    min_radius, max_radius = np.min(dist_to_barycenter), np.max(dist_to_barycenter)
    return shells(x_pc, min_radius + bins * (max_radius-min_radius), center=barycenter)

  def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, ensure_better: bool = False, **kwargs):
    from copy import deepcopy
    # print("Sample weight is None? ", sample_weight is None)
    assert isinstance(y, np.ndarray) and all(y >= 0), "y must be non-negative integers classes"
    sample_weight = np.ones(len(X))/len(X) if sample_weight is None else sample_weight
    assert np.isclose(np.sum(sample_weight), 1.0, atol=1e-8), f"Sample weight must be a distribution ({np.sum(sample_weight):8f} != 1 )."
    
    ## TODO: Sample from a precomputed set of large bins to be even faster?
    rng = np.random.RandomState(self.random_state) # used for sampling shells
    while not(hasattr(self, "bins_")) or (False if not(ensure_better) else self.score(X, y) <= 0.50):
      self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
      self.classes_, y = np.unique(y, return_inverse=True) # TODO: import from cache
      self.n_classes_ = len(self.classes_)                 # TODO: import from cache
      if not(hasattr(self, "cache_")) and self.cache:
        self.cache_ = {}

      ## Re-weight the sample weights to be a distribution on each class
      class_weights, norm_weight = {}, deepcopy(sample_weight)
      for cl in self.classes_: 
        class_weights[cl] = np.sum(norm_weight[y == cl])
        norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

      ## Build/fit a reference model
      self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
      if self.cache:
        x_hash = subsample_hash(X)
        if x_hash not in self.cache_:
          ## Precompute as much information as we can 
          self.cache_[x_hash] = {}
          for i, x in enumerate(X): 
            x_pc = x.reshape((len(x) // self.dim, self.dim))
            barycenter = x_pc.mean(axis=0)
            self.cache_[x_hash][i] = np.linalg.norm(x_pc - barycenter, axis=1)
        ccache = self.cache_[x_hash]
        for dist2center, yi, wi in zip(ccache.values(), y, norm_weight): 
          min_r, max_r = np.min(dist2center), np.max(dist2center)
          self.mean_curves[yi] += wi * np.histogram(dist2center, bins = min_r + self.bins_ * (max_r-min_r))[0]
      else:
        ## Build a weighted mean curve for each class to use for prediction
        for xi, yi, wi in zip(X, y, norm_weight): 
          self.mean_curves[yi] += wi * self.shells_signature(xi, self.bins_)

      # ## Preprocessing and randomization
      # rng = np.random.RandomState(self.random_state) # used for sampling shells
      # self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
      # self.classes_, y = np.unique(y, return_inverse=True) # scikit required 
      # self.n_classes_ = len(self.classes_)
      
      # ## Re-weight the sample weights to be a distribution on each class
      # class_weights, norm_weight = {}, deepcopy(sample_weight)
      # for cl in self.classes_: 
      #   class_weights[cl] = np.sum(norm_weight[y == cl])
      #   norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

      # ## Build a weighted mean curve for each class to use for prediction
      # self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
      # for x, label, w in zip(X, y, norm_weight): 
      #   self.mean_curves[label] += w * self.shells_signature(x, self.bins_)

      # ## Reverse the curves if its bad?
      # if self.score(X, y) < 0.50: 
      #   c0 = self.mean_curves[self.classes_[0]]
      #   c1 = self.mean_curves[self.classes_[1]]
      #   self.mean_curves[self.classes_[0]] = c1
      #   self.mean_curves[self.classes_[1]] = c0
      return self

  ## Needed for SAMME.R boosting
  def predict_proba(self, X: ArrayLike) -> ArrayLike:
    # https://stackoverflow.com/questions/4064630/how-do-i-convert-between-a-measure-of-similarity-and-a-measure-of-difference-di
    P = np.empty(shape=(X.shape[0], self.n_classes_))
    for i,x in enumerate(X):
      sig = self.shells_signature(x, self.bins_)
      d = np.array([np.sum(np.abs(curve-sig)) for curve in self.mean_curves.values()])
      s = np.exp(-d**1) # exponential conversion to similarity
      P[i,:] = s / np.sum(s)
    return P

  def predict(self, X: ArrayLike):
    P = self.predict_proba(X)
    return self.classes_[P.argmax(axis=1)]

  def score(self, X: ArrayLike, y: ArrayLike) -> float:
    return np.sum(self.predict(X) == y)/len(y)
  







class BarycenterClassifier(BaseEstimator, ClassifierMixin):
  """Classifier defined by distance to nearest class-fit barycenter."""
  def __init__(self, **kwargs):
    """sklearn requires no input validation for initializer """
    for k,v in kwargs.items():
      setattr(self, k, v)

  @staticmethod
  def vector(X: Any, **kwargs) -> np.ndarray:
    """Produces a vector embedding of a given input """
    raise NotImplementedError("A vector method must be implemented to use as an estimator.")
  
  @staticmethod
  def cdist(XA: ArrayLike, XB: ArrayLike, **kwargs) -> np.ndarray:
    """Compute distance between each pair of the two collections of inputs."""
    from scipy.spatial.distance import cdist
    return cdist(XA, XB, **kwargs)

  @staticmethod
  def barycenter(X: ArrayLike, sample_weight: ArrayLike = None, normalize: bool = True) -> np.ndarray:
    """Computes an representative average over a set of inputs."""
    if sample_weight is None:
      return X.mean(axis=0) 
    else: 
      assert isinstance(sample_weight, np.ndarray) and len(sample_weight) == len(X), "Sample weight must be array with a length matching X."
      sample_weight = sample_weight / np.sum(sample_weight) if normalize else sample_weight
      return (X @ np.diag(sample_weight)).mean(axis=0)

  def fit(self, X: List[Any],ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
    """Fits a set of weighted barycenters for each class.
    
    Populates barycenters_
    """
    self.classes_, y = np.unique(y, return_inverse=True)  # sklearn required 
    self.n_classes_ = len(self.classes_)                  # sklearn required 
    self.barycenters_ = { cl : self.barycenter(X[y == cl], sample_weight[y == cl] if not sample_weight is None else None) for cl in self.classes_ }
    return self

  def decision_function(self, X: ArrayLike) -> np.ndarray: 
    # V = np.array([self.vector(x) for x in X])
    V = self.vector(X)
    C = np.array([centroid for centroid in self.barycenters_.values()])
    return self.cdist(V, C)

  def predict_proba(self, X: ArrayLike) -> np.ndarray:
    """Converts distances to similarity (arbitrarily) to produce a probability."""
    # https://stackoverflow.com/questions/4064630/how-do-i-convert-between-a-measure-of-similarity-and-a-measure-of-difference-di
    P = self.decision_function(X)
    P = np.exp(-P) # exponential conversion to similarity: todo, generalize
    P = np.diag(1 / P.sum(axis=1)) @ P
    return P

  def predict(self, X: ArrayLike):
    """Bayes classifier."""
    P = self.predict_proba(X)
    return self.classes_[P.argmax(axis=1)]

  def score(self, X: ArrayLike, y: ArrayLike) -> float:
    """Accuracy as the default score."""
    return np.sum(self.predict(X) == y)/len(y)






def barycenter_classifier(name: str, vector: Callable, center: Callable = None, dist: Callable = None) -> BaseEstimator:
  """Constructs a basic classifier """
  methods = dict(vector=staticmethod(vector))
  if center is not None: 
    methods["barycenter"] = staticmethod(center)
  if dist is not None: 
    methods["cdist"] = staticmethod(dist)
  return type(name, (BarycenterClassifier,), methods)




from copy import deepcopy

## TODO: can this be generalized further by treating as a simplified bag-of-words type model
class AverageClassifierFactory(BaseEstimator, ClassifierMixin):
  """ Classifier factory for building learners that use distance to class-averages for classification """
  
  def __init__(self):

    pass 

  def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
    rng = np.random.RandomState(self.random_state) # used for sampling shells
    assert isinstance(y, np.ndarray) and all(y >= 0), "y must be non-negative integers classes"
    sample_weight = np.ones(len(X))/len(X) if sample_weight is None else sample_weight
    assert np.isclose(np.sum(sample_weight), 1.0, atol=1e-8), f"Sample weight must be a distribution ({np.sum(sample_weight):8f} != 1 )."
    
    ## TODO: Sample from a precomputed set of large bins to be even faster?
    
    self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
    self.classes_, y = np.unique(y, return_inverse=True) # TODO: import from cache
    self.n_classes_ = len(self.classes_)                 # TODO: import from cache
    if not(hasattr(self, "cache_")) and self.cache:
      self.cache_ = {}

    ## Re-weight the sample weights to be a distribution on each class
    class_weights, norm_weight = {}, deepcopy(sample_weight)
    for cl in self.classes_: 
      class_weights[cl] = np.sum(norm_weight[y == cl])
      norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

    ## Build/fit a reference model
    x_hash = self.subsample_hash(X)
    if self.cache:
      if x_hash not in self.cache_:
        ## Precompute as much information as we can 
        self.cache_[x_hash] = {}
        for i, x in enumerate(X): 
          x_pc = x.reshape((len(x) // self.dim, self.dim))
          barycenter = x_pc.mean(axis=0)
          self.cache_[x_hash][i] = np.linalg.norm(x_pc - barycenter, axis=1)
      else:
        self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
        ccache = self.cache_[x_hash]
        for dist2center, yi in zip(ccache.values(), y): 
          min_r, max_r = np.min(dist2center), np.max(dist2center)
          self.mean_curves[yi] += wi * np.histogram(dist2center, bins = min_r + self.bins_ * (max_r-min_r)) 
    else:
      ## Build a weighted mean curve for each class to use for prediction
      self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
      for xi, yi, wi in zip(X, y, norm_weight): 
        self.mean_curves[yi] += wi * self.shells_signature(xi, self.bins_)
      

    ## Reverse the curves if its bad?
    ## Issue: The spirit of "positive" and "negative" classification is sort of lost 
    # if self.score(X, y) < 0.50: 
    #   c0 = self.mean_curves[self.classes_[0]]
    #   c1 = self.mean_curves[self.classes_[1]]
    #   self.mean_curves[self.classes_[0]] = c1
    #   self.mean_curves[self.classes_[1]] = c0
    return self

  ## Needed for SAMME.R boosting
  def predict_proba(self, X: ArrayLike) -> ArrayLike:
    # https://stackoverflow.com/questions/4064630/how-do-i-convert-between-a-measure-of-similarity-and-a-measure-of-difference-di
    P = np.empty(shape=(X.shape[0], self.n_classes_))
    for i,x in enumerate(X):
      sig = self.shells_signature(x, self.bins_)
      d = np.array([np.sum(np.abs(curve-sig)) for curve in self.mean_curves.values()])
      s = np.exp(-d**1) # exponential conversion to similarity
      P[i,:] = s / np.sum(s)
    return P

  def predict(self, X: ArrayLike):
    P = self.predict_proba(X)
    return self.classes_[P.argmax(axis=1)]

  def score(self, X: ArrayLike, y: ArrayLike) -> float:
    return np.sum(self.predict(X) == y)/len(y)



# class BagOfFeatures(object):
#   def __init__(self, X: List[ArrayLike]):
#     self.X = X 

#   def fit(self, method=["kmeans", "uniform", "qke"], **kwargs):
#     from scipy.cluster.vq import kmeans2

#     self.prototypes_ = ...
  
#   def transform(self, X: List[ArrayLike], method=["voronoi", "qee", "qke"]) -> np.ndarray:
#     for V in X: 
#       np.argmin(cdist(V, self.prototypes_), axis=1)