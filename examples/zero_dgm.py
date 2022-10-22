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


nv = 150
X = scale_diameter(np.random.uniform(size=(nv,2)), 2.0)
X = X - (X.sum(axis=0)/X.shape[0])
K = delaunay_complex(X)
E, T = K['edges'], K['triangles']
E = E[np.sort(np.random.choice(range(E.shape[0]), size=int(nv/3), replace=False)),:]

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
 
E = np.array(list(cycle_window(range(len(fv)))))
dgm = ph0_lower_star(fv, E, max_death="max").astype(np.float32)
plot_dgm(dgm)

from pbsig.betti import lower_star_multiplicity
from pbsig.pht import rotate_S1
F = list(rotate_S1(X, 320, include_direction=False))
R = [[-1, 0, 0.07, 0.78], [0, 0.50, 0.51, 0.70]]

M = lower_star_multiplicity(F, E, R, max_death="max")

plt.plot(M[:,0])





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

