
from typing import * 
import numpy as np 
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator
import networkx as nx 
import primme
from scipy.sparse.linalg import * 
from scipy.sparse import * 
from pbsig.persistence import boundary_matrix
from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1, uniform_S1
from pbsig.datasets import mpeg7
from pbsig.pht import pht_preprocess_pc
from pbsig.simplicial import *

G = nx.connected_watts_strogatz_graph(n=30, k=5, p=0.10)
X = np.array(list(nx.fruchterman_reingold_layout(G).values()))


from pbsig.linalg import eigsh_family
from scipy.sparse.csgraph import laplacian

L = laplacian(nx.adjacency_matrix(G), form='lo')
deg = np.array([G.degree(i) for i in range(L.shape[0])])
L.diagonal = lambda: deg

K = {
  'vertices' : np.array(G.nodes()),
  'edges' : np.array(G.edges())
}


# L @ np.random.uniform(size=(L.shape[0], 1))
class Laplacian_DT_2D:
  def __init__(self, X: ArrayLike, K, nd: 132):
    self.theta = np.linspace(0, 2*np.pi, nd, endpoint=False)
    self.X = X 
    self.cc = 0
    self.D1 = boundary_matrix(K, p=1).tocsc()
    self.D1.sort_indices() # ensure !
    self.W = np.zeros(shape=(1,self.D1.shape[0]))

  def __iter__(self):
    self.cc = 0
    return self

  def __next__(self):
    if self.cc >= len(self.theta):
      raise StopIteration
    v = np.cos(self.theta[self.cc]), np.sin(self.theta[self.cc])
    self.W = diags(self.X @ v)
    self.cc += 1
    return self.W @ self.D1 @ self.D1.T @ self.W

  def __len__(self) -> int: 
    return len(self.theta)

L_dt = Laplacian_DT_2D(X, K, 32)

from pbsig.linalg import eigsh_family
results_lanczos = list(eigsh_family(L_iter, p = 1.0, reduce=sum, return_stats=True, tol=1e-6, return_history=True, method="PRIMME_Arnoldi",
  ncv = 0,             # Max basis size / Lanczos vectors to keep in-memory; use minimum three for 3-term recurrence 
  maxBlockSize = 0,    # Max num of vectors added at every iteration. This affects the num. of shifts given to the preconditioner
  minRestartSize = 0,  # Num. of approx. eigenvectors (Ritz vectors) kept during restart
  maxPrevRetain = 0    # Num. of approx. eigenvectors kept *from previous iteration* in restart; the +k in GD+k; called 'locally optimal restarting'
))
# results_jd = list(eigsh_family(L_iter, p = 0.80, reduce=sum, return_stats=True, method="PRIMME_JDQMR"))
# results_gd = list(eigsh_family(L_iter, p = 0.80, reduce=sum, return_stats=True, method="PRIMME_GD_Olsen_plusK"))
# results_sd = list(eigsh_family(L_iter, p = 0.80, reduce=sum, return_stats=True, method="PRIMME_STEEPEST_DESCENT"))


## The benchmakr plot 
import matplotlib.pyplot as plt
from matplotlib.pyplot import FixedLocator
fig = plt.figure(figsize=(4,4), dpi=180)
ax = fig.gca()

nv = len(K['vertices'])
x_ticks = np.cumsum(np.repeat(nv, len(results_lanczos)))
# fl = FixedLocator(np.cumsum(np.repeat(nv, len(results_lanczos))))
# ax.set_axes_locator(fl)

nmvs = [np.array(stats['hist']['numMatvecs'], dtype=int) for ew, stats in results_lanczos]
cum_mvs = 0
for i, mvs in enumerate(nmvs):
  cmvs = cum_mvs + np.cumsum(mvs-min(mvs))
  x_mv = np.linspace(i, i+1, len(cmvs))
  s = ax.scatter(cmvs, cmvs, c='blue', s=0.25)
  cum_mvs = max(cmvs)
ax.set_aspect("equal")
_ = plt.xticks(x_ticks, labels=[str(i) for i in range(len(results_lanczos))])
_ = plt.yticks(x_ticks, labels=[str(i) for i in range(len(results_lanczos))])

# ax.set_major_locator(fl)


np.log()

# results_lanczos[0][1]['hist'].keys()

results_lanczos



results_arnoldi[0][1]["hist"]["numMatvecs"]