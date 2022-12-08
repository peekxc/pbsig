import random
import numpy as np 
import networkx as nx 
from scipy.sparse import diags, eye 
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh
from scipy.sparse.csgraph import laplacian
from pbsig.simplicial import graph2complex
from pbsig.persistence import pHcol
from pbsig.betti import boundary_matrix

seed = 259
G = nx.connected_watts_strogatz_graph(n=15, k=5, p=0.10, seed=seed)
X = np.array(list(nx.fruchterman_reingold_layout(G, seed=seed).values()))
W = diags(X @ np.array([1,0]))
L = W @ laplacian(nx.adjacency_matrix(G)) @ W

import matplotlib.pyplot as plt 
fig, ax = plt.figure(figsize=(5,5)), plt.gca()
nx.draw(G, pos=X, ax=ax, node_size=18.5, node_color=W.diagonal())

ax.scatter(*X[[6,7,10],:].T, c='red', s=25.5, zorder=10)
ax.plot(X[[7,8],0], X[[7,8],1], c='red', linewidth=2.5, zorder=5)
ax.plot(X[[8,10],0], X[[8,10],1], c='red', linewidth=2.5, zorder=5)

# D1 = boundary_matrix(graph2complex(G), p = 1)
V, E = np.array(G.nodes()), np.array(G.edges())

from pbsig.persistence import barcodes
fv = W.diagonal()
fe = fv[E].max(axis=1)
ph0_f = barcodes(graph2complex(G), p = 0, f=(fv,fe), collapse=False, index=True)

FV, FE = V[np.argsort(fv)], E[np.argsort(fe),:]


fv_int = np.argsort(np.argsort(fv))
fe_int = fv_int[E].max(axis=1)
#fe_int = len(fv)+np.argsort(np.argsort(fe))
ph0_I = barcodes(graph2complex(G), p = 0, f=(fv_int, fe_int), collapse=False)
dgm0 = ph0_I['dgm']
dgm0[:,1] -= len(fv)

V = np.array(G.nodes())[np.argsort(fv)]
E = np.array(G.edges())[np.argsort(fe),:]


# np.array(G.edges())[np.argsort(fe)]

# from pbsig.persistence import ph0_lower_star
# dgm = ph0_lower_star(fv=W.diagonal(), E=np.array(G.edges()))
# #lower_star_ph_dionysus(W.diagonal(), E=np.array(G.edges()), T=[])


# I = np.argsort(np.argsort(W.diagonal()))
# ph0_lower_star(fv=I, E=np.array(G.edges()))

## Index persistence 
dgm0_fun = barcodes(graph2complex(G), p = 0, f=(fv,fe), collapse=False, index=False)
dgm0_ind = barcodes(graph2complex(G), p = 0, f=(fv,fe), collapse=False, index=True)

ns = G.number_of_nodes() + G.number_of_edges()

fig = plt.figure(figsize=(5,5))
ax = plt.gca()
major_ticks = np.arange(0, ns, 5)
minor_ticks = np.arange(0, ns, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(visible=True)
ax.set_ylim(0, ns)
ax.set_xlim(0, ns)
xb = dgm0_index[1:,0]
xd = dgm0_index[1:,1]+len(fv)
ax.scatter(xb, xd, c='red', s=6.5, zorder=10)
ax.set_aspect('equal')


from pbsig import plot_dgm
#plot_dgm(np.c_[dgm0_index[:,0], dgm0_index[:,1]+len(fv)])
plot_dgm(dgm0_fun['dgm'])


from itertools import chain
VL = [[v] for v in V]
EL = [list(e) for e in E]
filt_face_lex = (lambda S: (S[1], len(S[0]), S[0])) # simplex, simplex value
F = sorted(chain(zip(VL, fv), zip(EL, fe)), key=filt_face_lex)

## Make inverse map between f <-> 
F = { tuple(s) : f for s,f in F }
m = len(F)
fi,fj = -1.1, -0.5
fk,fl = -0.62, 1.0

filter_values = np.fromiter(iter(F.values()), float)
i = np.searchsorted(filter_values, fi, side='left')
j = np.searchsorted(filter_values, fj, side='right')-1
k = np.searchsorted(filter_values, fk, side='left')
l = np.searchsorted(filter_values, fl, side='right')-1

int(np.floor((i+j)/2))




vi = [i for i,(s,sf) in enumerate(F) if len(s) == 1]
ei = [i for i,(s,sf) in enumerate(F) if len(s) == 2]
dgm0 = barcodes(graph2complex(G), p = 0, f=(vi,ei), collapse=False, index=False)['dgm']

v_labels = np.array([F[i][0][0] for i in vi])



vi2 = np.array(vi)[np.argsort(v_labels)]

plot_dgm(dgm0)

np.arange(G.number_of_nodes())[np.argsort(W.diagonal())][0]

## Just do it 



