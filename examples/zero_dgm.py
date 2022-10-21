import numpy as np
from numpy.typing import ArrayLike
from itertools import combinations
from pbsig.utility import scale_diameter
from pbsig.simplicial import delaunay_complex
from pbsig import edges_from_triangles

nv = 12 
X = scale_diameter(np.random.uniform(size=(nv,2)), 2.0)
X = X - (X.sum(axis=0)/X.shape[0])
K = delaunay_complex(X)
E, T = K['edges'], K['triangles']
E = E[np.random.choice(range(E.shape[0]), size=6, replace=False),:]

## Plot the simplicial complex 
from pbsig import plot_mesh2D
fig, ax = plot_mesh2D(X, E, T, labels=True)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

def ph0_lower_star(fv: ArrayLike, E: ArrayLike, collapse: bool = True, lex_sort: bool = True) -> ArrayLike:
  from scipy.cluster.hierarchy import DisjointSet
  ## Data structures + variables
  nv = len(fv)
  ds = DisjointSet(range(nv)) # Set representatives are minimum by *index*
  elders = np.fromiter(range(nv), dtype=int) # Elder map: needed to maintain minimum set representatives
  paired = np.array([False]*nv, dtype=bool)
  insert_rule = lambda a,b: not(np.isclose(a,b)) if collapse else True # when to insert a pair
  
  ## Compute lower-star edge values; prepare to traverse in order
  fe = fv[E].max(axis=1)  # function evaluated on edges
  ei = np.fromiter(sorted(range(E.shape[0]), key=lambda i: (max(fv[E[i,:]]), min(fv[E[i,:]]))), dtype=int)
  
  ## Proceed to union components via elder rule
  dgm = []
  for (i,j), f in zip(E[ei,:], fe[ei]):
    ## The 'elder' was born first before the child: has smaller function value
    elder, child = (i, j) if fv[i] <= fv[j] else (j, i)
    # if elder == 5 and child == 1: raise ValueError("")
    if not ds.connected(i,j):
      if not paired[child]: # un-paired, dgm0[child][1] == np.inf
        dgm += [(fv[child], f)] if insert_rule(fv[child], f) else []
        paired[child] = True # kill the child
        #print(f"{child}, ({i},{j})")
      else: # child already paired, use elder rule (keep elder alive)
        creator = elders[elder] if fv[elders[child]] <= fv[elders[elder]] else elders[child]
        dgm += [(fv[creator], f)] if insert_rule(fv[creator], f) else []
        paired[creator] = True
        #print(f"{creator}, ({i},{j})")
      
      ## Representative step: get set-representatives w/ minimum function value
      elder_i, elder_j = elders[ds[i]], elders[ds[j]]
      elders[i] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j
      elders[j] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j
      
      ## Merge (i,j) and update elder map
      ds.merge(i,j)
      
  ## Post-processing 
  # print(f"Essential: {str(np.flatnonzero(paired == False))}")
  eb = fv[np.flatnonzero(paired == False)]  ## essential births
  ep = list(zip(eb, [np.inf]*len(eb)))      ## essential pairs 
  dgm = np.array(dgm + ep)                  ## append essential pairs

  ## If warranted, lexicographically sort output
  return dgm if not(lex_sort) else dgm[np.lexsort(np.rot90(dgm)),:]

nv = 15004
X = scale_diameter(np.random.uniform(size=(nv,2)), 2.0)
X = X - (X.sum(axis=0)/X.shape[0])
K = delaunay_complex(X)
E, T = K['edges'], K['triangles']
E = E[np.sort(np.random.choice(range(E.shape[0]), size=int(nv/30), replace=False)),:]

# from pbsig import plot_mesh2D
# fig, ax = plot_mesh2D(X, E, T, labels=True)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)

fv = (X @ np.array([0,1])).flatten()
#ph0_lower_star(fv, E).astype(np.float32)

dgm0_truth = lower_star_ph_dionysus(fv, E, [])[0]
dgm0_truth = dgm0_truth[np.lexsort(np.rot90(dgm0_truth)),:]
dgm0_test = ph0_lower_star(fv, E).astype(np.float32)

x = dgm0_test - dgm0_truth
assert np.allclose(x[~np.isnan(x)], 0.0)
 








import networkx as nx
G = nx.gnp_random_graph(32, 0.05)
X = np.array([p for p in nx.fruchterman_reingold_layout(G).values()])
E = np.array(list(G.edges))

v = np.array([0,1])

fe = fv[E].max(axis=1)

nx.draw(G, pos=X, with_labels=True)

dgm0_truth = lower_star_ph_dionysus(fv, E, [])[0]

#dgm0 = dgm0[np.lexsort(np.rot90(dgm0)),:]
dgm0_truth[np.lexsort(np.rot90(dgm0_truth)),:]
ph0_lower_star(fv, E)

abs(dgm0 - dgm0_truth)



## PHT idea 
v = np.array([0,1])
fv = (X @ v[:,np.newaxis]).flatten()
fe = fv[E].max(axis=1)

from pbsig.persistence import lower_star_ph_dionysus
lower_star_ph_dionysus(fv, E, [])[0]

