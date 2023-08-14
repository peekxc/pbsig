from itertools import *
from typing import * 
from numpy.typing import ArrayLike

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.cluster.vq import kmeans2, vq, whiten

from .linalg import logsample
from .shape import shells

def subsample_hash(X: np.ndarray, size: int = 100) -> int:
  """Hashes the byte representation of a numpy array by random-sampling."""
  rng = np.random.RandomState(89)
  inds = rng.randint(low=0, high=X.size, size=size)
  b = X.flat[inds]
  b.flags.writeable = False
  return hash(b.data.tobytes())

class BinaryAdaBoostClassifier:
  """Binary AdaBoost classifier."""
  def __init__(self, weak_learner: object, random_state = None):
    self.random_state = random_state
    self.weak_learner = weak_learner

  def fit(self, X: ArrayLike, y: ArrayLike, n_estimators: int = 100, cap_loss: bool = True, normalize_weights: bool = True):
    """Fits a binary boosted classifier. 
    
    Capping the exponential loss to [0,1] is equivalent to using a 0/1 mistake "dichotomy". If False, the loss 
    for misclassification can be arbitrarily high ([0, \infty)), which could lead to faster convergence at 
    the cost of numerical instability. 

    Constraining the sample weights to sum to 1 via weight normalization is common in the literature. 
    This has the effect of passing a measure / distribution of samples to the weak learners to fit too, and 
    it internally implies the error / learning coefficient update will derive from an expectation taken over 
    the hypothesis space of weak learners. Moreover, if the weak learners .fit() function creates ground truth 
    information using these sample weights (e.g. in the form of barycenters), then normalizing the weight vector
    effectively restricts the hypothesis space of those weak learners to the convex hull of each classes training data.
    
    In the context of barycentric coordinates, normalizing the weight vector allows a sample-weight-fitted estimator
    to have a unique hypothesis space, as the corresponding barycenters are homogenous. 
    
    When normalize_weights = True, AdaBoost can be seen as maintaining a growth relation between the sum of KL
    divergences between successive weight vectors and our expected value of of the margins. This has a dual interpretation 
    of finding support vectors that minimize the margin as the number of learners grows, see https://arxiv.org/pdf/2210.07808.pdf.

    The unique values and dtype of the y array will determine the output of any function that returns class labels (e.g. predict) 

    Parameters: 
      X: ndarray representing points in Euclidean space. 
      y: ndarray of class labels. See details. 
      n_estimators: the number of weak learners to fit.
      cap_loss: whether to restrict the exponential loss to lie in [0,1]. See details. 
      normalize_weights: whether to normalize the sample coefficients to sum to 1. See details. 
    """
    from copy import deepcopy
    n = X.shape[0]
    classes_, y = np.unique(y, return_inverse=True)
    assert n == len(y) and isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "Invalid (X, y) pair given."
    assert len(classes_) == 2, "AdaBoost is only a binary classifier"
    self.classes_ = classes_
    self.estimators_ = []
    self.coeffs_ = []

    ## Run the iterations
    ## Update formula: https://course.ccs.neu.edu/cs6140sp15/4_boosting/lecture_notes/boosting/boosting.pdf
    ## Also: https://arxiv.org/pdf/2210.07808.pdf
    ## rt = 1.0-et
    ## np.where(yt != y, wt * np.sqrt(rt/et), wt * np.sqrt(et/rt))  ## alternative, equiv weight update
    ## 0.5 * np.log((1+(1-2*et))/(1 - (1-2*et)))                    ## alternative, equiv coefficient calculation
    ## Compare with: https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/ensemble/weight_boosting.py#L572-L574
    ## sklearn uses 0/1 loss w/ no wieght normalization, though normalization may be necessary: https://stats.stackexchange.com/questions/59490/the-weight-updating-in-adaboost
    wt = np.ones(n) / n                                   ## uniform coefficient weights 
    self.weak_learner.fit(X, y)                           ## fit the first estimator with (X,y) pair - subsequent .fit() calls should be ignored
    for t in range(1, n_estimators):
      ht = self.weak_learner.fit(X, y, wt)                ## fit weak learner with weighted samples; should be randomized
      yt = ht.predict(X)                                  ## predict using the weak learner
      et = np.sum(wt[yt != y]) / np.sum(wt)               ## error of the current weak learner; strictly non-negative
      at = 0.5 * np.log((1.0-et)/et) if et > 0 else 1.0   ## voting/learning coefficient      ; in (-\infty, 1]
      nt = np.where(yt == y, 1, -1)                       ## signed "mistake dichotomy" -- could also use 0/1 loss to cap mistake contributions
      wt *= np.exp(-at * nt)                              ## weight update; exp(*) >= 0 ==> wt >= 0
      wt /= np.sum(wt)                                    ## weight normalization; optional, though weights should stay non-negative
      self.estimators_.append(deepcopy(ht))
      self.coeffs_.append(at)

  def decision_function(self, X: ArrayLike) -> ArrayLike: 
    """Confidence function."""
    y_pred = np.zeros(len(X), dtype=np.float64)
    sgn_classes = np.array([-1, 1], dtype=np.int16) 
    for a, h in zip(self.coeffs_, self.estimators_):
      p = np.searchsorted(self.classes_, h.predict(X)) # guarenteed to be in [0,1]
      y_pred += a * sgn_classes[p] # maps class @ 0 -> -1 and class @ 1 -> 1
    return y_pred

  def predict_proba(self, X: ArrayLike) -> ArrayLike:
    """Probability prediction function."""
    y_pred = self.decision_function(X)
    m = np.sum(np.abs(self.coeffs_)) # all predictions should be in [-m, m] where m = sum(alpha) = max confidence
    prob = 0.5 + np.where(y_pred < 0, (m - np.abs(y_pred))/(2*m), y_pred/(2*m))
    return np.array([(p, 1-p) if s < 0 else (1-p, p) for p,s in zip(prob, y_pred)])
    # normalize = lambda x: (x - np.min(x))/max_confidence
    # proba = (y_pred + max_confidence)/(2*max_confidence) # should be in [0,1]

  def predict(self, X: ArrayLike) -> ArrayLike:
    return self.classes_[np.where(A.decision_function(X) < 0, 0, 1)]
  
  def margin(self, X: ArrayLike, y: ArrayLike):
    """Computes the margin of classifier success or 'confidence' on a given (X,y) pair."""
    sgn_classes = np.array([-1, 1], dtype=np.int16) 
    yp = sgn_classes[np.searchsorted(self.classes_, y)]
    return self.decision_function(X) * yp

  def staged_decision_function(self, X: ArrayLike) -> Generator:
    y_pred = np.zeros(len(X), dtype=np.float64)
    sgn_classes = np.array([-1, 1], dtype=np.int16) 
    for a, h in zip(self.coeffs_, self.estimators_):
      p = np.searchsorted(self.classes_, h.predict(X)) # guarenteed to be in [0,1]
      y_pred += a * sgn_classes[p] # maps class @ 0 -> -1 and class @ 1 -> 1
      yield deepcopy(y_pred)

  def staged_predict_proba(self, X: ArrayLike) -> Generator:
    m = np.sum(np.abs(self.coeffs_)) # max confidence
    for y_pred in self.staged_decision_function(X):
      prob = 0.5 + np.where(y_pred < 0, (m - np.abs(y_pred))/(2*m), y_pred/(2*m))
      yield np.array([(p, 1-p) if s < 0 else (1-p, p) for p,s in zip(prob, y_pred)])
      
  def staged_predict(self, X: ArrayLike) -> Generator:
    for y_pred in self.staged_decision_function(X):
      yield self.classes_[np.where(y_pred < 0, 0, 1)]

  def staged_margin(self, X: ArrayLike, y: ArrayLike, normalize: bool = False) -> Generator:
    margin = np.zeros(len(y), dtype=np.float64)
    for i, (yt, at) in enumerate(zip(self.staged_predict(X), self.coeffs_)):
      margin += at * np.where(yt == y, 1, -1)
      yield deepcopy(margin/np.sum(np.abs(self.coeffs_[:(i+1)]))) if normalize else deepcopy(margin)

  def score(self, X: ArrayLike, y: ArrayLike) -> float:
    return np.sum(self.predict(X) == y)/len(y)

  def staged_score(self, X: ArrayLike, y: ArrayLike) -> Generator:
    for yt in self.staged_predict(X):
      yield float(np.sum(yt == y)/len(y))

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
  """Classifier defined by distance to nearest class-fit barycenter.
  
  The default behavior fits class-specific barycenters and yields an classifier whose decision_function depends on the chosen metric. 
  """
  def __init__(self, metric: Union[str, Callable] = "euclidean", random_state = None, **kwargs):
    """sklearn requires no input validation for initializer """
    self.metric = metric # to be called by cdist
    self.random_state = random_state 
    for k,v in kwargs.items():
      setattr(self, k, v)

  @staticmethod
  def vector(X: Any, **kwargs) -> np.ndarray:
    """Produces a vector embedding of a given input.
    
    Defaults to the identity function (w/ input validation).
    """
    # raise NotImplementedError("A vector method must be implemented to use as an estimator.")
    return np.asarray_chkfinite(X)
  
  # @staticmethod
  # def cdist(XA: ArrayLike, XB: ArrayLike, **kwargs) -> np.ndarray:
  #   """Compute distance between each pair of the two collections of inputs."""
  #   from scipy.spatial.distance import cdist
  #   return cdist(XA, XB, **kwargs)

  @staticmethod
  def barycenter(X: ArrayLike, sample_weight: ArrayLike = None) -> np.ndarray:
    """Computes an weighted average over a set of inputs."""
    if sample_weight is None:
      return X.mean(axis=0) 
    else: 
      ## Use barycentric coordinates
      assert isinstance(sample_weight, np.ndarray) and len(sample_weight) == len(X), "Sample weight must be array with a length matching X."
      return (sample_weight[:,np.newaxis] * X).mean(axis=0)

  def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
    """Fits a set of weighted barycenters for each class.
    
    Populates barycenters_
    """
    self.classes_, y = np.unique(y, return_inverse=True)  # sklearn required 
    self.n_classes_ = len(self.classes_)                  # sklearn required 
    # [x for xi, yi in zip(X, y) if yi == cl]
    # X_cl = lambda cl: [x for x, yi in zip(X, y) if yi == cl]
    self.barycenters_ = { cl : self.barycenter(X[y == cl], sample_weight[y == cl] if not sample_weight is None else None) for cl in self.classes_ }
    return self

  def decision_function(self, X: ArrayLike) -> np.ndarray:
    """ Computes the distance between vectors of 'X' the class-fitted barycenters"""
    from scipy.spatial.distance import cdist
    # V = np.array([self.vector(x) for x in X])
    V = self.vector(X)
    C = np.array([centroid for centroid in self.barycenters_.values()])
    return cdist(V, C, metric=self.metric)

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


def barycenter_classifier(name: str, center: Callable = None, vector: Callable = None) -> BaseEstimator:
  """Constructs a barycenter classifier using the supplied vector and centering functions. 
  """
  methods = {}
  if center is not None: 
    methods["barycenter"] = staticmethod(center)
  if vector is not None:
    methods["vector"] = staticmethod(vector)
  return type(name, (BarycenterClassifier,), methods)


from copy import deepcopy
from scipy.stats import uniform

## TODO: Cretae a generalized Ensemble Learner or a Hypothesis space Classifier that uses subsample hashing on 
## fit, predict (+ related methods) to cache fixed training inputs 
## TODO: THEN, create a generalized base estimator that accounts for knowing correspondences
## Feature extractors must implement at least: fit, transform, get_feature_names_out
## https://scikit-learn.org/stable/glossary.html#term-feature-extractor
## Transformers must be estimators supporting transform and/or fit_transform
## for warm_start, see: https://scikit-learn.org/stable/glossary.html#term-warm_start
class HeatKernelClassifier(BaseEstimator, ClassifierMixin): # _VectorizerMixin
  """ Constructs a meta-estimator from a heat kernel instance. """
  def __init__(self, heat_kernel, dimension: int, distribution = uniform, metric: str = "euclidean", random_state = None, **kwargs): 
    self.heat_kernel = heat_kernel 
    self.time_interval = heat_kernel.time_bounds("absolute") # can be changed
    self.random_state = random_state
    self.distribution = distribution
    self.dimension = dimension
    self.metric = metric

  def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None, sample_weight: ArrayLike = None):
    """Fits a classifier to a signature produced by the underlying heat kernel at a randomly sampled timepoint. 
    """
    if not(hasattr(self, "X_hash_")):
      # assert card(self.complex,0) == X.shape[0]
      self.X_hash_ = subsample_hash(X)
      self.heat_kernels_ = []
      for x in X: 
        P = x.reshape(len(x) // self.dimension, self.dimension)
        hk = self.heat_kernel.clone().fit(X=P, approx="mesh", normed=False)
        self.heat_kernels_.append(hk)
    else: 
      assert (self.X_hash_ == subsample_hash(X)), "This estimator can only be used on a fixed training set."
    
    ## Sample new timepoint(s)
    ## "If, for some reason, randomness is needed after fit, the RNG should be stored in an attribute random_state_"
    self.distribution.random_state = self.random_state
    prop = self.distribution.rvs(size=1)
    self.timepoints_ = logsample(*self.time_interval, num=prop) # time for diffusion 
    
    ## Fit to class-specific signatures using either bag-of-feature vector quantization technique 
    ## or class-specific barycenter idea (w/ generalized procrustes). If the latter, it's assumed the 
    ## correspondence problem has already been solved, such that each row in X is already aligned. 
    self.signature_ = np.array([np.ravel(hk.signature(self.timepoints_)) for hk in self.heat_kernels_])
    self.estimator_ = BarycenterClassifier(metric=self.metric) # metric should be signal_dist
    self.estimator_.fit(self.signature_, y, sample_weight)

    ## If X is ArrayLike, it's assumed fixed number of points, fixed dimension, and correspondence problem is solved 
    ## In which case we can use simple barycenter based classifier or generalized procrustes for |t| > 1
    ## - The dimension problem can be resolved w/ a iterable of inputs w/ a given shape 
    ## - The fixed number of points can be alleviated using landmarks (to yield a fixed # of points)
    ## - Correspondence problem can be solved either: 
    ##    a. Solve the global correspondence for every pair via supervision or *trusted* procrustes 
    ##    b. Solve it per-pair by modding out via cross-correlation. Only works with 1-parameter  
    return self
    
  def transform(self, X: ArrayLike, y: ArrayLike = None):
    """ y is unused. """
    if hasattr(self, "X_hash_") and self.X_hash_ == subsample_hash(X):
      return self.signature_
    else: 
      hks = []
      for x in X: 
        P = x.reshape(len(x) // self.dimension, self.dimension)
        hks.append(self.heat_kernel.clone().fit(P, approx="mesh", normed=False))
      sigs = np.array([np.ravel(hk.signature(self.timepoints_)) for hk in hks])
      return sigs
    
  def decision_function(self, X: ArrayLike, *args, **kwargs) -> np.ndarray:     
    Xt = self.transform(X)
    return self.estimator_.decision_function(Xt, *args, **kwargs)
  
  def predict_proba(self, X: ArrayLike, *args, **kwargs) -> np.ndarray: 
    Xt = self.transform(X)
    return self.estimator_.predict_proba(Xt, *args, **kwargs)
  
  def predict(self, X: ArrayLike, *args, **kwargs) -> np.ndarray: 
    Xt = self.transform(X)
    return self.estimator_.predict(Xt, *args, **kwargs)

  def score(self, X: ArrayLike, *args, **kwargs) -> float:
    Xt = self.transform(X)
    return self.estimator_.score(Xt, *args, **kwargs)



# ## TODO: can this be generalized further by treating as a simplified bag-of-words type model
# class AverageClassifierFactory(BaseEstimator, ClassifierMixin):
#   """ Classifier factory for building learners that use distance to class-averages for classification """
  
#   def __init__(self):

#     pass 

#   def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None, **kwargs):
#     rng = np.random.RandomState(self.random_state) # used for sampling shells
#     assert isinstance(y, np.ndarray) and all(y >= 0), "y must be non-negative integers classes"
#     sample_weight = np.ones(len(X))/len(X) if sample_weight is None else sample_weight
#     assert np.isclose(np.sum(sample_weight), 1.0, atol=1e-8), f"Sample weight must be a distribution ({np.sum(sample_weight):8f} != 1 )."
    
#     ## TODO: Sample from a precomputed set of large bins to be even faster?
    
#     self.bins_ = self.bins if isinstance(self.bins, np.ndarray) else np.sort(rng.uniform(low=0, high=1.0, size=self.bins+1))
#     self.classes_, y = np.unique(y, return_inverse=True) # TODO: import from cache
#     self.n_classes_ = len(self.classes_)                 # TODO: import from cache
#     if not(hasattr(self, "cache_")) and self.cache:
#       self.cache_ = {}

#     ## Re-weight the sample weights to be a distribution on each class
#     class_weights, norm_weight = {}, deepcopy(sample_weight)
#     for cl in self.classes_: 
#       class_weights[cl] = np.sum(norm_weight[y == cl])
#       norm_weight[y == cl] = norm_weight[y == cl] / class_weights[cl] if class_weights[cl] > 0 else norm_weight[y == cl]

#     ## Build/fit a reference model
#     x_hash = self.subsample_hash(X)
#     if self.cache:
#       if x_hash not in self.cache_:
#         ## Precompute as much information as we can 
#         self.cache_[x_hash] = {}
#         for i, x in enumerate(X): 
#           x_pc = x.reshape((len(x) // self.dim, self.dim))
#           barycenter = x_pc.mean(axis=0)
#           self.cache_[x_hash][i] = np.linalg.norm(x_pc - barycenter, axis=1)
#       else:
#         self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
#         ccache = self.cache_[x_hash]
#         for dist2center, yi in zip(ccache.values(), y): 
#           min_r, max_r = np.min(dist2center), np.max(dist2center)
#           self.mean_curves[yi] += wi * np.histogram(dist2center, bins = min_r + self.bins_ * (max_r-min_r)) 
#     else:
#       ## Build a weighted mean curve for each class to use for prediction
#       self.mean_curves = { cl : np.zeros(len(self.bins_)-1) for cl in self.classes_ }
#       for xi, yi, wi in zip(X, y, norm_weight): 
#         self.mean_curves[yi] += wi * self.shells_signature(xi, self.bins_)
      

#     ## Reverse the curves if its bad?
#     ## Issue: The spirit of "positive" and "negative" classification is sort of lost 
#     # if self.score(X, y) < 0.50: 
#     #   c0 = self.mean_curves[self.classes_[0]]
#     #   c1 = self.mean_curves[self.classes_[1]]
#     #   self.mean_curves[self.classes_[0]] = c1
#     #   self.mean_curves[self.classes_[1]] = c0
#     return self

#   ## Needed for SAMME.R boosting
#   def predict_proba(self, X: ArrayLike) -> ArrayLike:
#     # https://stackoverflow.com/questions/4064630/how-do-i-convert-between-a-measure-of-similarity-and-a-measure-of-difference-di
#     P = np.empty(shape=(X.shape[0], self.n_classes_))
#     for i,x in enumerate(X):
#       sig = self.shells_signature(x, self.bins_)
#       d = np.array([np.sum(np.abs(curve-sig)) for curve in self.mean_curves.values()])
#       s = np.exp(-d**1) # exponential conversion to similarity
#       P[i,:] = s / np.sum(s)
#     return P

#   def predict(self, X: ArrayLike):
#     P = self.predict_proba(X)
#     return self.classes_[P.argmax(axis=1)]

#   def score(self, X: ArrayLike, y: ArrayLike) -> float:
#     return np.sum(self.predict(X) == y)/len(y)


class BagOfFeatures(object):
  def __init__(self, X: List[ArrayLike]):
    self.X = X 

  def fit(self, k: int, method: str = "kmeans", **kwargs):
    # ["kmeans", "uniform", "qke"]
    p = { 'iter': 1, 'minit': "++"} | kwargs
    if method == "kmeans":
      centroids, _ = kmeans2(whiten(self.X), k=k, **p) 
    elif method == "uniform":
      centroids = np.random.choice(len(self.X), size=k, replace=False)
    elif method == "qke":
      raise NotImplementedError("Haven't implemented yet")
    else:
      raise ValueError("Invalid BoF methododology")
    self.prototypes_ = centroids
    return self # "The method should return the object (self)"

  # "even unsupervised estimators need to accept a y=None keyword argument in the second position that is just ignored by the estimator."
  def transform(self, X: List[ArrayLike], y: ArrayLike = None, method=["voronoi", "qee", "qke"]) -> np.ndarray:
    bof = np.zeros((len(X), len(self.prototypes_)), dtype=np.uint16)
    for i, V in enumerate(X): 
      codebook_labels, centroid_dist = vq(whiten(V), self.prototypes_) # Assign codes from a code book to observations.
      np.add.at(bof[i,:], codebook_labels, 1)
    return bof
