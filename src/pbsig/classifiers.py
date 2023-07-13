from itertools import *
from typing import * 
from numpy.typing import ArrayLike
import numpy as np
from pbsig.shape import shells
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class ShellsClassifier(BaseEstimator, ClassifierMixin):
  """ Weak classifier that uses a histogram of distances to perform classification """
  def __init__(self, bins: Union[int, Iterable] = 10, dim: int = 2, random_state = None):
    self.random_state = random_state # np.random.RandomState(random_state) # used for sampling shells
    self.bins = bins
    self.dim = dim  # ambient dimension of the point cloud 
    
  def shells_signature(self, x: ArrayLike, bins: Optional[ArrayLike] = None) -> ArrayLike:
    bins = self.bins_ if bins is None else bins
    x_pc = x.reshape((len(x) // self.dim, self.dim))
    barycenter = x_pc.mean(axis=0)
    dist_to_barycenter = np.linalg.norm(x_pc - barycenter, axis=1)
    min_radius, max_radius = np.min(dist_to_barycenter), np.max(dist_to_barycenter)
    return shells(x_pc, min_radius + bins * (max_radius-min_radius), center=barycenter)

  def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
    from copy import deepcopy
    # print("Sample weight is None? ", sample_weight is None)
    assert isinstance(y, np.ndarray) and all(y >= 0), "y must be non-negative integers classes"
    sample_weight = np.ones(len(X))/len(X) if sample_weight is None else sample_weight
    assert np.isclose(np.sum(sample_weight), 1.0, atol=1e-8), f"Sample weight must be a distribution ({np.sum(sample_weight):8f} != 1 )."
    
    ## Preprocessing and randomization
    rng = np.random.RandomState(self.random_state) # used for sampling shells
    self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
    self.classes_, y = np.unique(y, return_inverse=True) # scikit required 
    self.n_classes_ = len(self.classes_)
    
    ## Re-weight the sample weights to be a distribution on each class
    class_weights, norm_weight = {}, deepcopy(sample_weight)
    for cl in self.classes_: 
      class_weights[cl] = np.sum(norm_weight[y == cl])
      norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

    ## Build a weighted mean curve for each class to use for prediction
    self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
    for x, label, w in zip(X, y, norm_weight): 
      self.mean_curves[label] += w * self.shells_signature(x, self.bins_)

    ## Reverse the curves if its bad?
    if self.score(X, y) < 0.50: 
      c0 = self.mean_curves[self.classes_[0]]
      c1 = self.mean_curves[self.classes_[1]]
      self.mean_curves[self.classes_[0]] = c1
      self.mean_curves[self.classes_[1]] = c0
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
  

class ShellsFactory(BaseEstimator, ClassifierMixin):
  """ Weak classifier factory that improves the computational efficiency of SHELLS on a fixed data set """
  _precomputed = {}
  
  def __init__(self, bins: Union[int, Iterable] = 10, dim: int = 2, random_state = None):
    self.random_state = random_state # np.random.RandomState(random_state) # used for sampling shells
    self.bins = bins
    self.dim = dim  # ambient dimension of the point cloud 
    
  def shells_signature(self, x: ArrayLike, bins: Optional[ArrayLike] = None) -> ArrayLike:
    bins = self.bins_ if bins is None else bins
    x_pc = x.reshape((len(x) // self.dim, self.dim))
    barycenter = x_pc.mean(axis=0)
    dist_to_barycenter = np.linalg.norm(x_pc - barycenter, axis=1)
    min_radius, max_radius = np.min(dist_to_barycenter), np.max(dist_to_barycenter)
    return shells(x_pc, min_radius + bins * (max_radius-min_radius), center=barycenter)
  
  def subsample_hash(self, X: np.ndarray, size: int = 100):
    rng = np.random.RandomState(89)
    inds = rng.randint(low=0, high=X.size, size=size)
    b = X.flat[inds]
    b.flags.writeable = False
    return hash(b.data)

  def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
    from copy import deepcopy
    # print("Sample weight is None? ", sample_weight is None)
    assert isinstance(y, np.ndarray) and all(y >= 0), "y must be non-negative integers classes"
    sample_weight = np.ones(len(X))/len(X) if sample_weight is None else sample_weight
    assert np.isclose(np.sum(sample_weight), 1.0, atol=1e-8), f"Sample weight must be a distribution ({np.sum(sample_weight):8f} != 1 )."
    rng = np.random.RandomState(self.random_state) # used for sampling shells
    
    x_hash = self.subsample_hash(X)
    if x_hash not in self._precomputed:
      ## Precompute as much information as we can 
      pass 
    else:
      ## Sample from a precomputed set of large bins
      
      self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
      self.classes_, y = np.unique(y, return_inverse=True) # TODO: import from cache
      self.n_classes_ = len(self.classes_)                 # TODO: import from cache
    
      ## Re-weight the sample weights to be a distribution on each class
      class_weights, norm_weight = {}, deepcopy(sample_weight)
      for cl in self.classes_: 
        class_weights[cl] = np.sum(norm_weight[y == cl])
        norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

    ## Build a weighted mean curve for each class to use for prediction
    self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
    for x, label, w in zip(X, y, norm_weight): 
      self.mean_curves[label] += w * self.shells_signature(x, self.bins_)

    ## Reverse the curves if its bad?
    if self.score(X, y) < 0.50: 
      c0 = self.mean_curves[self.classes_[0]]
      c1 = self.mean_curves[self.classes_[1]]
      self.mean_curves[self.classes_[0]] = c1
      self.mean_curves[self.classes_[1]] = c0
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