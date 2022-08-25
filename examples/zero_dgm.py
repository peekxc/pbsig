import numpy as np
from itertools import combinations
from pbsig.utility import scale_diameter
from pbsig.simplicial import delaunay_complex
from pbsig import edges_from_triangles

# n = 8
# X = np.random.uniform(low=0, high=1, size=(n,2))
# E = []
# p = 0.2
# for i,j in combinations(range(n), 2):
#   if np.random.random() <= p:
#     E.append([i,j])


n = 8 
X = np.random.uniform(size=(8,2))
X = scale_diameter(X, 2.0)
X = X - (X.sum(axis=0)/X.shape[0])
K = delaunay_complex(X)
E, T = K['edges'], K['triangles']

## Plot the simplicial complex 
fig, ax = plot_mesh2D(X, K['edges'], K['triangles'], labels=True)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

from scipy.cluster.hierarchy import DisjointSet

v = np.array([0,1])
fv = (X @ v[:,np.newaxis]).flatten()
fe = fv[E].max(axis=1)

elders = np.fromiter(range(len(fv)), dtype=int)
dgm0 = { vi : (f, np.inf) for vi,f in enumerate(fv) }
ds = DisjointSet(range(len(fv)))
e_ind = np.argsort(fe)
for (i,j), f in zip(E[e_ind,:], fe[e_ind]):
  elder, child = (i, j) if fv[i] <= fv[j] else (j, i)
  if not ds.connected(i,j):
    # if child == 7 and elder == 6: 
    #   raise ValueError("")
    if dgm0[child][1] == np.inf: #un-paired
      dgm0[child] = (fv[child], f)
    else: # child already paired, use elder rule
      creator = elders[child] if fv[elders[child]] <= fv[elders[elder]] else elders[elder]
      dgm0[creator] = (fv[creator], f)
    elder_i, elder_j = elders[ds[i]], elders[ds[j]]
    ds.merge(i,j)
    #p = ds[i] # merged parent of i and j
    elders[i] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j
    elders[j] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j
    #elders[p] = elders[pi] if fv[elders[pi]] <= fv[elders[pj]] else elders[pj] # elders[1] = elders[6] = 1, elders[7]=3
    # else:
    #   child = elders[elder]
    #   dgm0[child] = (fv[child], f)
    #   ds.merge(i,j)
    #   elders[ds[i]] = elders[ds[i]] = elder 
    #   print("")

nx.draw(G, pos=X, with_labels=True)
# v_ind = np.argsort(fv)
# for v, f in zip(v_ind, fv[v_ind]):

import networkx as nx
G = nx.gnp_random_graph(8, 0.3)
X = np.array([p for p in nx.fruchterman_reingold_layout(G).values()])
E = np.array(list(G.edges))

v = np.array([0,1])
fv = (X @ v[:,np.newaxis]).flatten()


from pbsig.persistence import lower_star_ph_dionysus
lower_star_ph_dionysus(fv, E, [])


nx.draw(G, pos=X, with_labels=True)
