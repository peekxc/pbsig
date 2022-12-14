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

from pbsig.persistence import barcodes
from pbsig.simplicial import * 
fv = W.diagonal()
K = graph2complex(G)
K = SimplicialComplex(chain(chain(iter(K['vertices']), iter(K['edges'])), iter(K['triangles'])))
F = MutableFiltration(K, f=lambda s: fv[s].max())
ph0_f = barcodes(F)

# FV, FE = V[np.argsort(fv)], E[np.argsort(fe),:]
# fv_int = np.argsort(np.argsort(fv))
# fe_int = fv_int[E].max(axis=1)
# #fe_int = len(fv)+np.argsort(np.argsort(fe))
# ph0_I = barcodes(graph2complex(G), p = 0, f=(fv_int, fe_int), collapse=False)
# dgm0 = ph0_I['dgm']
# dgm0[:,1] -= len(fv)

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
fk,fl = -0.49, 1.0

fig, ax = plot_dgm(dgm0_fun['dgm'])
ax.plot([fi,fj],[fk,fk],linewidth=0.25)
ax.plot([fi,fj],[fl,fl],linewidth=0.25)
ax.plot([fi,fi],[fk,fl],linewidth=0.25)
ax.plot([fj,fj],[fk,fl],linewidth=0.25)


boundary_matrix(K)

## Build full filtered boundary matrix
from array import array 
Sgn = { 1 : np.array([1, -1]), 2: np.array([1, -1, 1]) }
s_hashes = [hash(Simplex(s)) for s, fs in F.items()]
m = len(F)
ri,ci,dv = array('I'), array('I'), array('f')
for i, s in enumerate(F.keys()):
  s = Simplex(s)
  if s.dimension() >= 1:
    ind = np.sort([s_hashes.index(hash(b)) for b in s.boundary()])
    ri.extend(ind)
    ci.extend(np.repeat(i, len(ind)))
    dv.extend(Sgn[s.dimension()])
D = coo_array((dv, (ri,ci)), shape=(m,m))

filter_values = np.fromiter(iter(F.values()), float)
i = np.searchsorted(filter_values, fi, side='left')
j = np.searchsorted(filter_values, fj, side='right')-1
k = np.searchsorted(filter_values, fk, side='left')
l = np.searchsorted(filter_values, fl, side='right')-1


D = D.tocsc()

# persistence_pairs(R)
np.array(list(filter(lambda x: x != -1, low_entry(R))))


from scipy.sparse import coo_array
IJ = np.array([(j,i) for i,j in enumerate(low_entry(R)) if j != -1])
S = coo_array((np.repeat(1, len(IJ)), (IJ[:,0],IJ[:,1])), shape=R.shape).tocsc()

plt.spy(S, markersize=3.5)
plt.spy(R, markersize=3.5)

n = S.shape[0]
S[i:n,1:(k+1)]

def mu_query(D,i,j,k,l):
  A1,A2,A3,A4 = D[i:(l+1),i:(l+1)], D[i:k,i:k], D[(j+1):(l+1),(j+1):(l+1)], D[(j+1):k,(j+1):k]
  #assert A1.shape == ((l-i+1), (l-i+1))
  t1 = 0 if A1.shape == (0,0) else np.linalg.matrix_rank(A1.todense())
  t2 = 0 if A2.shape == (0,0) else np.linalg.matrix_rank(A2.todense())
  t3 = 0 if A3.shape == (0,0) else np.linalg.matrix_rank(A3.todense())
  t4 = 0 if A4.shape == (0,0) else np.linalg.matrix_rank(A4.todense())
  return t1 - t2 - t3 + t4
  #np.linalg.matrix_rank(D[i:(j+1),k:(l+1)].todense())
mu_query(D,i,j,k,l)

def bisection_tree_top_down(M,rank_f,ind,mu,splitrows=True):
  i,j,k,l = ind 
  if mu == 1 and (i == j if splitrows else k == l):
    piv = i if splitrows else k
    print(f"[{mu}]: ({i},{j},{k},{l}): creator = {i}, destroyer = {k} ({piv})")
    yield piv
  else:
    p = int(np.floor((i+j)/2)) if splitrows else int(np.floor((k+l)/2)) # pivot
    ind_lc = (i,p,k,l) if splitrows else (i,j,k,p)
    ind_rc = (p+1,j,k,l) if splitrows else (i,j,p+1,l)
    mu_lc = rank_f(*ind_lc) #np.linalg.matrix_rank(M[i:(piv+1),k:(l+1)].todense())
    mu_rc = rank_f(*ind_rc) #np.linalg.matrix_rank(M[(piv+1):(j+1),k:(l+1)].todense())
    print(f"[{mu}]: ({i},{j},{k},{l}), L:{mu_lc}, R:{mu_rc}")
    assert mu == mu_lc + mu_rc
    if mu_lc > 0:
      yield from bisection_tree_top_down(M,rank_f,ind_lc,mu_lc,splitrows)
    if mu_rc > 0:
      yield from bisection_tree_top_down(M,rank_f,ind_rc,mu_rc,splitrows)

mu = mu_query(i,j,k,l)
creators = list(bisection_tree_top_down(D,mu_query,(i,j,k,l),mu,splitrows=True))

creators = list(bisection_tree_top_down(D,mu_query,(0,0,k,l),1,splitrows=False))
list(bisection_tree_top_down(D,mu_query,(1,1,k,l),splitrows=False))
list(bisection_tree_top_down(D,mu_query,(2,2,k,l),splitrows=False))

creators = []
bisection_tree(D,i,j,k,l,creators)

destroyers = []
bisection_tree2(D,2,2,k,l,destroyers)
# for c in creators:
  


mu_ij = np.linalg.matrix_rank(D[i:(j+1),k:(l+1)].todense())
piv = int(np.floor((i+j)/2))
mu_lc = np.linalg.matrix_rank(D[i:(piv+1),k:(l+1)].todense())
mu_rc = np.linalg.matrix_rank(D[piv+1:(j+1),k:(l+1)].todense())
assert mu_lc + mu_rc == mu_ij


vi = [i for i,(s,sf) in enumerate(F) if len(s) == 1]
ei = [i for i,(s,sf) in enumerate(F) if len(s) == 2]
dgm0 = barcodes(graph2complex(G), p = 0, f=(vi,ei), collapse=False, index=False)['dgm']

v_labels = np.array([F[i][0][0] for i in vi])



vi2 = np.array(vi)[np.argsort(v_labels)]

plot_dgm(dgm0)

np.arange(G.number_of_nodes())[np.argsort(W.diagonal())][0]

## Just do it 



## binary search 
def binary_search(a, P: Callable, lo=0, hi=len(a)): 
  """ imitates bisection, doing binary search until a given predicate P(a) is True """
  if P(a[lo]):
    return lo
  else:
    piv = int(np.floor(lo+hi))
    return binary_search(a, P, piv, hi)

## TODO: fix via additivity 
def bisection_tree(M,i,j,k,l,results):
  if i > j: 
    return 0
  elif i == j:
    mu_ij = np.linalg.matrix_rank(M[i:(j+1),k:(l+1)].todense())
    if mu_ij == 1:
      results.append(i) # is a creator
    return mu_ij
  else:
    assert j-i > 0
    piv = int(np.floor((i+j)/2))
    print(f"{i},{piv},{j}")
    mu_lc = bisection_tree(M,i,piv,k,l,results)
    mu_rc = bisection_tree(M,piv+1,j,k,l,results)
    return mu_lc + mu_rc
    
def bisection_tree2(M,i,j,k,l,results):
  if k > l: 
    return 0
  elif k == l:
    mu_ij = np.linalg.matrix_rank(M[i:(j+1),k:(l+1)].todense())
    if mu_ij == 1:
      results.append(k) # is a creator
    return mu_ij
  else:
    assert l-k > 0
    piv = int(np.floor((l+k)/2))
    print(f"{k},{piv},{l}")
    mu_lc = bisection_tree2(M,i,j,k,piv,results)
    mu_rc = bisection_tree2(M,i,j,piv+1,l,results)
    return mu_lc + mu_rc