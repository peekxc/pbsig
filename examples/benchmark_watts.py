
from typing import * 
from numpy.typing import ArrayLike

import numpy as np 
import networkx as nx 
import primme

from scipy.sparse.linalg import * 
from scipy.sparse import * 
from scipy.sparse.csgraph import laplacian
from pbsig.persistence import boundary_matrix
from pbsig.linalg import lanczos
from pbsig.simplicial import cycle_graph
from pbsig.pht import rotate_S1, uniform_S1
from pbsig.pht import pht_preprocess_pc
from pbsig.simplicial import *
from pbsig.linalg import eigsh_family

## Data set 
G = nx.watts_strogatz_graph(n=500, k=10, p=0.15)
S = simplicial_complex(G.edges())
k = int(np.ceil(np.log2(124749)))
er = rank_combs(S.faces(1), n=500)

number = 1337
with open('complex.txt', 'w') as f:
  for r in er:
    f.write('%d\n' % r)
# ./build -n 18625 -c 3.0 -a 0.94 -e dictionary_dictionary --minimal -i /Users/mpiekenbrock/pbsig/examples/complex.txt

# X = np.array(list(nx.fruchterman_reingold_layout(G).values()))

# edge_bs = 0b0001
# y = np.array([(edge_bs << k) | r for r in er])
# z = np.array([int(f[0]) for f in S.faces(0)])
# np.array(list(z) + list(y))


rank_combs(S.faces(1), k=2, n=500)

## Form weighted Laplacian directional transform 
from pbsig.betti import Laplacian_DT_2D

L = laplacian(nx.adjacency_matrix(G))
# deg = np.array([G.degree(i) for i in range(L.shape[0])])
# L.diagonal = lambda: deg

K = {
  'vertices' : np.array(G.nodes()),
  'edges' : np.array(G.edges())
}
L_dt = Laplacian_DT_2D(X, K, 32)
L = next(iter(Laplacian_DT_2D(X, K, 32)))

## Default PRIMME parameters
primme_opts = dict(
  return_stats=True, tol=1e-6, return_history=True, 
  #maxiter=int(L.shape[0]*40), 
  method="PRIMME_Arnoldi",  # The specialization of the Generalized Davidson to use
  ncv = 3,                  # Max basis size / Lanczos vectors to keep in-memory; use minimum three for 3-term recurrence 
  maxBlockSize = 0,         # Max num of vectors added at every iteration. This affects the num. of shifts given to the preconditioner
  minRestartSize = 0,       # Num. of approx. eigenvectors (Ritz vectors) kept during restart
  maxPrevRetain = 0         # Num. of approx. eigenvectors kept *from previous iteration* in restart; the +k in GD+k; called 'locally optimal restarting'
) 
eigsh_opts = dict(
  p = 0.50, reduce=sum
)

from pbsig.linalg import eigsh_family
opts = primme_opts | { 'ncv' : 20 }
res_lanczos = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_Arnoldi"))))              # non-locking + no inner method
res_jd = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_JDQMR"))))                     # non-locking + "robust shifts" preconditioner via QMR
res_gd = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_GD_Olsen_plusK"))))        # non-locking + projector for correction equation; uses locally optimal restarting
# results_sd = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_STEEPEST_DESCENT"))))      # locking + projector for correction equation
res_lobpcg = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_LOBPCG_OrthoBasis_Window")))) # non-locking + locally optimal preconditioned CG 
res_dyn = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_DYNAMIC"))))  

## The residual norm plot
import matplotlib.pyplot as plt
from pbsig.color import bin_color
mvs = np.array(res_lanczos[0][1]['hist']['numMatvecs'])
ren = np.array(res_lanczos[0][1]['hist']['resNorm'])
nec = np.array(res_lanczos[0][1]['hist']['nconv'])
#rgb = bin_color()
# plt.plot(mvs, nec)

## Estimate convergence rate
# https://www.math-cs.gordon.edu/courses/ma342/handouts/rate.pdf
# from pbsig.utility import window
# from pbsig.utility import pairwise
# seq_counts = np.cumsum(np.append([-1], np.bincount(nec)))+1
# alpha, ns = 0, 0
# for i,j in pairwise(seq_counts):
#   if abs(j-i) > 4:
#     #break
#     rn = np.array(ren[i:j])
#     plt.plot(np.array(ren[i:j]))
#     plt.yscale("log")
#     alpha += np.mean([np.log(abs((x4-x3)/(x3-x2)))/np.log(abs((x3-x2)/(x2-x1))) for x1,x2,x3,x4 in window(rn, 4)])
#     ns += 1

# ## Mean convergence rate    
# alpha/ns


## Alternate basis sizes
from pbsig.utility import timeout
res_memory = {}
for ncv in range(3, 30, 1):
  for p in [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]:
    try: 
      #timeout
      _, stats = list(timeout(eigsh_family, args=[[L]], kwargs=(opts | dict(ncv=ncv, p=p)), timeout_duration=5))[0]
      #_, stats = list(eigsh_family([L], **opts))[0]
    except Exception as e:
      stats = [int(L.shape[0]*40)]
    finally: 
      res_memory[(ncv,p)] = stats
    print(ncv)

## All four plots: 
## 1. (geometry) Embedding of data set, with nodes and edges highlighted by function value 
## 2. (memory) Lanczos (matvecs/n) x ncv for ncv = 3...30 ( justifies O(c*n) memory usage, where c is a small constant)
## 3. (time) Plot num. converged eigenvalues (x axis) vs (matvecs/n) (y-axis) to get idea of how ratios needed by each approach to get 1-eps of spectrum 
## 4. (stability) Stability type plot over directional transform to show speed-up from using previous results (perturbation expansion?)
import matplotlib.pyplot as plt
from pbsig.color import bin_color

fig, axs = plt.subplots(1, 4, figsize=(15,3), dpi=250)

## Geometry plot 
fv = X @ np.array([0,1])
v_rgb, e_rgb = bin_color(fv), bin_color(fv[K['edges']].max(axis=1))
axs[0].scatter(*X.T, c=v_rgb, zorder=10)
for cc, (i,j) in enumerate(K['edges']):
  axs[0].plot(X[[i,j],0], X[[i,j],1], c=e_rgb[cc]) 
axs[0].set_title("Random Watts-Strogatz graph")
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_xlabel(f"Parameters: N={100}, k={5}, p={0.10}")
# axs[0].axis('off')

## Memory plot
for p in [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]:
  p_keys = list(filter(lambda k: k[1] == p, list(res_memory.keys())))
  axs[1].plot([res_memory[key]['numMatvecs']/L.shape[0] for key in p_keys], label=f"p={int(p*100)}%")

axs[1].legend(loc='upper right', prop={'size': 8})
axs[1].set_xlabel('Maximum basis size')
axs[1].set_ylabel('Matvecs/n')
axs[1].set_title("Memory Useage", fontsize=10)

## Time plot 
mvs_per_ew = lambda x: np.cumsum(np.bincount(x[0][1]['hist']['nconv']))/L.shape[0]

axs[2].plot(mvs_per_ew(res_lanczos), label="Lanczos")
axs[2].plot(mvs_per_ew(res_jd), label="Jacobi Davidson (+QMR)") #  inexact inner iteration is stopped dynamically and optimally
axs[2].plot(mvs_per_ew(res_gd), label="Gen. Davidson (+k)") #  recurrence based restarting; 
#plt.plot(mvs_per_ew(res_lobpcg), label="LOBPCG")
axs[2].plot(mvs_per_ew(res_dyn), label="Dynamic JD+GD")
axs[2].legend(loc='upper left', prop={'size': 8})
axs[2].set_ylabel('Matvecs/n')
axs[2].set_xlabel('Num. converged eigenvalues')
axs[2].set_title("Method Performance @ p=50%", fontsize=10)
axs[2].set_ylim(0.0, 4.0)


axs[3].plot(np.array(HIST)/L.shape[0], label="Cold start")
axs[3].plot(np.array(HIST_V0)/L.shape[0], label="Warm start")
axs[3].set_title("Full spectrum Directional Transform", fontsize=10)
axs[3].set_ylabel('Matvecs/n')
axs[3].set_xlabel('Index of consecutive Laplacian')
axs[3].legend(loc='upper right', prop={'size': 8})
axs[3].set_ylim(np.floor(min(min(np.array(HIST)/L.shape[0]), min(np.array(HIST_V0)/L.shape[0])))-1, np.ceil(max(np.array(HIST)/L.shape[0])*1.2))


L_dt = Laplacian_DT_2D(X, K, 132)
HIST_V0, HIST = [], []
v0 = None
eigsh_opts['p'] = 1.0
for L in L_dt:
  res_jd_local_fresh = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_JDQMR", return_eigenvectors=False, v0=None, return_history=False))))
  res_jd_local_v0 = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_JDQMR", return_eigenvectors=True, v0=v0, return_history=False))))
  v0 = res_jd_local_v0[0][1]
  HIST_V0.append(res_jd_local_v0[0][2]['numMatvecs'])
  HIST.append(res_jd_local_fresh[0][1]['numMatvecs'])



L_dt = Laplacian_DT_2D(X, K, 132)
eigsh_opts['p'] = 1.0
eigsh_opts['reduce'] = lambda x : x
W = []
for L in L_dt: 
  res_lanczos = list(eigsh_family([L], **eigsh_opts, **(opts | dict(method="PRIMME_Arnoldi"))))    
  ew = res_lanczos[0][0]
  W.append(ew)

plt.plot(np.cumsum(W[0]))
np.quantile(W[0], [0.05, 0.10, 0.25, 0.50, 0.75, 1.0])

EV = W[0]
cs = np.cumsum(EV) 
Q = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
I = [np.flatnonzero(cs >= q)[0] for q in Q*sum(EV)]

plt.plot(EV)
for cc,(i, q) in enumerate(zip(I,Q)):
  plt.vlines(i, 0, 3, colors='red', linestyles='dotted')
  if cc % 2 == 0:
    plt.text(i, 3.1, f"{int(q*100)}%")
  else:
    plt.text(i, -0.1, f"{int(q*100)}%", ha='center')

plt.ylim(-0.5, 3.4)
plt.title("Eigenvalues (+quantiles) of L in descending order")

EPS_S = ["1e-8", "1e-6", "1e-4", "1e-2", "0.1", "0.5", "1", "5"]
for eps, s in zip([1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1, 5], EPS_S): #np.linspace(0.0001, 5.0, 10):
  plt.plot(EV/(EV + eps), label=f"eps={s}")
plt.legend()
plt.title("Behavior of eps on Laplacian eigenvalues")
plt.xlabel("Eigenvalue index")
plt.ylabel("Contribution to Rank relaxation")


  


## estimate lower bound for the rank 
L = next(iter(Laplacian_DT_2D(X, K, 32)))
rank_lb = np.ceil(L.trace()**2 / np.sum(L.data**2))

## Just seems unreliable
L = next(iter( Laplacian_DT_2D(X, K, 32)))
RN = [] # np.zeros(L.shape)
V0 = L @ np.random.uniform(size=(L.shape[0], L.shape[0]-1))
for i in range(V0.shape[1]): V0[:,i] / np.linalg.norm(V0[:,i])
for t in np.linspace(1.0, 1e-6, 100):
  results_single = list(eigsh_family([L], p = 1.0, reduce=sum, return_stats=True, tol=t, return_history=False,
    v0=V0,
    method="PRIMME_Arnoldi",  # The specialization of the Generalized Davidson to use
    ncv = 3,                  # Max basis size / Lanczos vectors to keep in-memory; use minimum three for 3-term recurrence 
    maxBlockSize = 0,         # Max num of vectors added at every iteration. This affects the num. of shifts given to the preconditioner
    minRestartSize = 0,       # Num. of approx. eigenvectors (Ritz vectors) kept during restart
    maxPrevRetain = 0,         # Num. of approx. eigenvectors kept *from previous iteration* in restart; the +k in GD+k; called 'locally optimal restarting', 
    return_eigenvectors=True
  ))
  V0 = results_single[0][1]
  RN.append(results_single[0][2]['rnorms'])

# W = np.vstack(RN)
# plt.plot(W[2,:])
# plt.plot(W[3,:])

def nmvs(L):
  return [primme.eigsh(L, return_stats=True, tol=1e-6, return_history=False,
      k=5,
      which="SM",
      v0 = np.c_[np.repeat(1/L.shape[0], L.shape[0])],
      sigma=0,
      method="PRIMME_GD_Olsen_plusK",  # The specialization of the Generalized Davidson to use
      maxBlockSize = 0,         # Max num of vectors added at every iteration. This affects the num. of shifts given to the preconditioner
      minRestartSize = 0,       # Num. of approx. eigenvectors (Ritz vectors) kept during restart
      maxPrevRetain = 0,
      ncv = ncv,
      return_eigenvectors=False
    )[1]['numMatvecs'] for ncv in range(3, 120, 5)]
plt.plot(nmvs(L))

W = [nmvs(L) for L in L_dt]


plt.plot(results_lanczos[0][1]['hist']['nconv'])
plt.plot(len(results_lanczos[0][1]['hist']['numMatvecs']))


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