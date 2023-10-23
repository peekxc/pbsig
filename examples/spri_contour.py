import numpy as np 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

from pbsig.betti import MuFamily
from pbsig.linalg import up_laplacian
from pbsig.datasets import random_lower_star
from pbsig.persistence import ph
from pbsig.vis import figure_complex
from itertools import chain, product, combinations
from pbsig.betti import smooth_upstep, smooth_dnstep
from pbsig.linalg import stable_rank

X, K = random_lower_star(130)
show(figure_complex(K, pos=X))


triangles = list(faces(K, 2))
tri_ind = np.random.choice(range(card(K,2)), size=int(card(K,2)*0.70), replace=False)
S = simplicial_complex(chain(faces(K,0), faces(K,1), [triangles[i] for i in tri_ind]))

diam = max(pdist(X))
LO = up_laplacian(S, p=0, form='lo', normed=True)
solver = eigvalsh_solver(LO)

fv = X @ np.array([0,1])
f = lower_star_filter(fv)
fe = np.array([f(s) for s in faces(S, 1)])

f_ticks = np.linspace(-diam, diam, 10)
results = {}
for i,j in combinations(f_ticks, 2):
  if i < j: 
    si = smooth_upstep(lb=i, ub=i)
    sj = smooth_dnstep(lb=j, ub=j)
    LO.set_weights(si(fv),sj(fe),si(fv))
    d = np.sqrt(pseudoinverse(LO.degrees))
    LO.set_weights(d, sj(fe), d)
    results[(i,j)] = spectral_rank(solver(LO))

DX = np.array([(x,y) for x,y in results.keys()])

from pbsig.color import bin_color

p = figure(width=200, height=200, x_range=(-diam, diam), match_aspect=True)
p.scatter(*DX.T, size=5, color=bin_color(list(results.values())))
show(p)

def rank_ij(c): 
  i,j = c
  si = smooth_upstep(lb=i, ub=i)
  sj = smooth_dnstep(lb=j, ub=j)
  LO.set_weights(si(fv),sj(fe),si(fv))
  d = np.sqrt(pseudoinverse(LO.degrees))
  LO.set_weights(d, sj(fe), d)
  return spectral_rank(solver(LO))

## Data to contour is the sum of two Gaussian functions.
x_rng = -diam, diam
y_rng = -diam, diam
xc, yc = np.meshgrid(np.linspace(*x_rng, 40), np.linspace(*y_rng, 40))
z = np.array([rank_ij((i,j)) for i,j in zip(np.ravel(xc), np.ravel(yc))]).reshape(xc.shape)

## Show decision boundary
p = figure(width=300, height=300, match_aspect=True, aspect_ratio=1)
levels = np.linspace(np.min(z), np.max(z), 30)
contour_r = p.contour(xc, yc, z, levels, line_color="black")
# p.scatter(X[:,0], X[:,1], color=["red"]*int(m/2) + ["blue"]*int(m/2))
show(p)





solver(LO)






K = filtration(S)

K.reindex(lower_star_filter(X @ np.array([1,0])))
dgm_v0 = ph(K, engine="dionysus")
K.reindex(lower_star_filter(X @ np.array([0.5,0.5])))
dgm_v1 = ph(K, engine="dionysus")


import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

bottleneck_distance(dgm_v0[1], dgm_v1[1])
wasserstein_distance(dgm_v0[1]['birth'], dgm_v1[1]['birth'])

def bottleneck_distance(D1, D2):
  """
  Compute the bottleneck distance between two persistence diagrams.
  
  Parameters:
      D1: Persistence diagrams w/ birth and death times.
      D2: 
  
  Returns:
      bd (float): Bottleneck distance between D1 and D2.
  """
  # Compute the distance matrix between points in D1 and D2
  Dmat = cdist(np.c_[D1['birth']], np.c_[D2['birth']])
  # Dmat = cdist(D1, D2)

  # Initialize matching dictionaries and distances
  matches1 = {}
  matches2 = {}
  dists1 = {i: np.inf for i in range(len(D1))}
  dists2 = {j: np.inf for j in range(len(D2))}

  # Find the best matching pairs using the Hungarian algorithm
  row_idx, col_idx = linear_sum_assignment(Dmat)
  for i, j in zip(row_idx, col_idx):
    dist = Dmat[i, j]
    if dist < dists1[i]:
      if i in matches1:
        del matches2[matches1[i]]
      matches1[i] = j
      matches2[j] = i
      dists1[i] = dist
      dists2[j] = dist
  return np.max(np.fromiter(dists1.values(), dtype=float) + np.fromiter(dists2.values(), dtype=float))






