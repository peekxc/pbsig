import random
import numpy as np 
import networkx as nx 
from itertools import chain
from scipy.sparse import diags, eye 
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh
from scipy.sparse.csgraph import laplacian
from pbsig.persistence import pHcol, barcodes
from pbsig.betti import boundary_matrix
from pbsig.simplicial import * 

seed = 259
G = nx.connected_watts_strogatz_graph(n=15, k=5, p=0.10, seed=seed)
X = np.array(list(nx.fruchterman_reingold_layout(G, seed=seed).values()))
W = diags(X @ np.array([1,0]))
L = W @ laplacian(nx.adjacency_matrix(G)) @ W


fv = W.diagonal()
K = MutableFiltration(map(Simplex, chain(G.nodes, G.edges)), f=lambda s: fv[s].max())
DGM = barcodes(K)


filter_values = np.fromiter(iter(K.keys()), float)
fi,fj,fk,fl = np.quantile(filter_values, [0.2, 0.4, 0.6, 0.8])
i = np.searchsorted(filter_values, fi, side='left')
j = np.searchsorted(filter_values, fj, side='right')-1
k = np.searchsorted(filter_values, fk, side='left')
l = np.searchsorted(filter_values, fl, side='right')-1

from pbsig.betti import mu_query
from pbsig.linalg import up_laplacian
L = up_laplacian(K, p=0, weight=lambda s: fv[s].max() + 1.0, form="lo")
mu_query(L, (i,j,k,l), f=lambda s: fv[s].max() + 1.0)

L.simplices

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

# from pbsig.vis import plot_complex

# import matplotlib.pyplot as plt 
# fig, ax = plt.figure(figsize=(5,5)), plt.gca()
# nx.draw(G, pos=X, ax=ax, node_size=18.5, node_color=W.diagonal())

# ax.scatter(*X[[6,7,10],:].T, c='red', s=25.5, zorder=10)
# ax.plot(X[[7,8],0], X[[7,8],1], c='red', linewidth=2.5, zorder=5)
# ax.plot(X[[8,10],0], X[[8,10],1], c='red', linewidth=2.5, zorder=5)
# K = SimplicialComplex(chain(chain(iter(K['vertices']), iter(K['edges'])), iter(K['triangles'])))
# D1 = boundary_matrix(graph2complex(G), p = 1)





# fig = plt.figure(figsize=(5,5))
# ax = plt.gca()
# major_ticks = np.arange(0, ns, 5)
# minor_ticks = np.arange(0, ns, 1)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)
# ax.grid(visible=True)
# ax.set_ylim(0, ns)
# ax.set_xlim(0, ns)
# xb = dgm0_index[1:,0]
# xd = dgm0_index[1:,1]+len(fv)
# ax.scatter(xb, xd, c='red', s=6.5, zorder=10)
# ax.set_aspect('equal')


from pbsig import plot_dgm
#plot_dgm(np.c_[dgm0_index[:,0], dgm0_index[:,1]+len(fv)])
plot_dgm(dgm0_fun['dgm'])


# from itertools import chain
# VL = [[v] for v in V]
# EL = [list(e) for e in E]
# filt_face_lex = (lambda S: (S[1], len(S[0]), S[0])) # simplex, simplex value
# F = sorted(chain(zip(VL, fv), zip(EL, fe)), key=filt_face_lex)



# fig, ax = plot_dgm(dgm0_fun['dgm'])
# ax.plot([fi,fj],[fk,fk],linewidth=0.25)
# ax.plot([fi,fj],[fl,fl],linewidth=0.25)
# ax.plot([fi,fi],[fk,fl],linewidth=0.25)
# ax.plot([fj,fj],[fk,fl],linewidth=0.25)


# boundary_matrix(K)

## Build full filtered boundary matrix
# from array import array 
# Sgn = { 1 : np.array([1, -1]), 2: np.array([1, -1, 1]) }
# s_hashes = [hash(Simplex(s)) for s, fs in F.items()]
# m = len(F)
# ri,ci,dv = array('I'), array('I'), array('f')
# for i, s in enumerate(F.keys()):
#   s = Simplex(s)
#   if s.dimension() >= 1:
#     ind = np.sort([s_hashes.index(hash(b)) for b in s.boundary()])
#     ri.extend(ind)
#     ci.extend(np.repeat(i, len(ind)))
#     dv.extend(Sgn[s.dimension()])
# D = coo_array((dv, (ri,ci)), shape=(m,m))




# D = D.tocsc()

# # persistence_pairs(R)
# np.array(list(filter(lambda x: x != -1, low_entry(R))))


# from scipy.sparse import coo_array
# IJ = np.array([(j,i) for i,j in enumerate(low_entry(R)) if j != -1])
# S = coo_array((np.repeat(1, len(IJ)), (IJ[:,0],IJ[:,1])), shape=R.shape).tocsc()

# plt.spy(S, markersize=3.5)
# plt.spy(R, markersize=3.5)

# n = S.shape[0]
# S[i:n,1:(k+1)]

# def mu_query(D,i,j,k,l):
#   A1,A2,A3,A4 = D[i:(l+1),i:(l+1)], D[i:k,i:k], D[(j+1):(l+1),(j+1):(l+1)], D[(j+1):k,(j+1):k]
#   #assert A1.shape == ((l-i+1), (l-i+1))
#   t1 = 0 if A1.shape == (0,0) else np.linalg.matrix_rank(A1.todense())
#   t2 = 0 if A2.shape == (0,0) else np.linalg.matrix_rank(A2.todense())
#   t3 = 0 if A3.shape == (0,0) else np.linalg.matrix_rank(A3.todense())
#   t4 = 0 if A4.shape == (0,0) else np.linalg.matrix_rank(A4.todense())
#   return t1 - t2 - t3 + t4
#   #np.linalg.matrix_rank(D[i:(j+1),k:(l+1)].todense())
# mu_query(D,i,j,k,l)


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