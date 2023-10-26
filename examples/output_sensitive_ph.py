import random
import numpy as np 
import networkx as nx 
from itertools import chain
from scipy.sparse import diags, eye 
from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh
from scipy.sparse.csgraph import laplacian
from pbsig.persistence import *
from pbsig.betti import boundary_matrix
from pbsig.simplicial import * 
import _persistence as pm
from splex import *
from bokeh.plotting import show, figure 
from bokeh.io import output_notebook
output_notebook()

## Test output sensitive algorithm 
S = simplicial_complex([[0,1,2], [0,1,3]])
K = filtration(S)
D = boundary_matrix(K)
R, V = D.copy().tolil(), eye(len(K)).tolil()
pHcol(R, V)
assert is_reduced(R)
dgm = generate_dgm(K, R)

from pbsig.vis import figure_dgm
show(figure_dgm(dgm[0]))

## Cone complex 
K = filtration(S)
K.add(ValueSimplex([-1], value=-np.inf)) # dummy vertex
max_weight = max(list(K.indices())) + 1
for s in faces(K):
  K.add(ValueSimplex(list(s) + [-1], value=max_weight)) # should this be inf?

## Now recompute diagram
# dgm_coned = ph(K)
# dgm_coned = { p : np.fromiter(iter(filter(lambda pair: pair[0] != -np.inf, dgm)), dtype=dgm.dtype) for p, dgm in dgm_coned.items() }
# show(figure_dgm(dgm_coned[0]))

## Convert to np.ndarray
B = boundary_matrix(K).todense()
# B = R.todense()
rrank = lambda i,j,k,l: 0 if len(B[i:j,k:l]) == 0 else np.linalg.matrix_rank(B[i:j,k:l])
mu_rank = lambda i,j,k,l: rrank(i,l+1,i,l+1) - rrank(i,k,i,k) - rrank(j+1,l+1,j+1,l+1) + rrank(j+1,k,j+1,k)
ind = (0, 3, 3, 7)

## Check every low-left matrix 
mu_rank(0,5,5,10) 
mu_rank(0,5,5,10) 
mu_rank(7,7,7,9)  # pivot for H1! 

def bisection_tree_top_down(mu_q: Callable, ind: tuple, mu: int, splitrows: bool = True, verbose: bool = False):
  assert len(ind) == 4 and tuple(sorted(ind)) == ind, "Invalid box given. Must be in upper-half plane."
  i,j,k,l = ind 
  if mu == 1 and (i == j if splitrows else k == l):
    piv = i if splitrows else k
    if verbose: print(f"[{mu}]: ({i},{j},{k},{l}): creator = {i}, destroyer = {k} ({piv})")
    yield piv
  else:
    p = int(np.floor((i+j)/2)) if splitrows else int(np.floor((k+l)/2)) # pivot
    ind_lc = (i,p,k,l) if splitrows else (i,j,k,p)
    ind_rc = (p+1,j,k,l) if splitrows else (i,j,p+1,l)
    mu_lc = mu_q(*ind_lc) #np.linalg.matrix_rank(M[i:(piv+1),k:(l+1)].todense())
    mu_rc = mu_q(*ind_rc) #np.linalg.matrix_rank(M[(piv+1):(j+1),k:(l+1)].todense())
    if verbose: print(f"[{mu}]: ({i},{j},{k},{l}), L:{mu_lc}, R:{mu_rc}")
    assert mu == mu_lc + mu_rc
    if mu_lc > 0:
      yield from bisection_tree_top_down(mu_q,ind_lc,mu_lc,splitrows)
    if mu_rc > 0:
      yield from bisection_tree_top_down(mu_q,ind_rc,mu_rc,splitrows)

## First find creators, then identify each corresponding destroyer in the unique pair with bisection again
dgm_res = []
creators = list(bisection_tree_top_down(mu_rank, ind=ind, mu=mu_rank(*ind), splitrows=True))
for i in creators:
  d = list(bisection_tree_top_down(mu_rank, ind=(i, i, ind[2], ind[3]), mu=mu_rank(i, i, ind[2], ind[3]), splitrows=False))
  assert len(d) == 1
  dgm_res.append([i, d[0]])
print(dgm_res)

## TODO: implement either SLICING/subsetting or MASKING for a linear operator, also checks for nnz, todense, etc
def pairs_in_box(box: tuple):
  assert len(box) == 4 and tuple(sorted(box)) == box
  dgm_res = []
  creators = list(bisection_tree_top_down(mu_rank, ind=box, mu=mu_rank(*box), splitrows=True))
  for idx in creators:
    d = list(bisection_tree_top_down(mu_rank, ind=(idx, idx, box[2], box[3]), mu=mu_rank(idx, idx, box[2], box[3]), splitrows=False))
    assert len(d) == 1
    dgm_res.append([idx, d[0]])
  return dgm_res 

## Now, come up with set of mu query windows that completely recover the diagram
## The idea: start with finding all pairs (a,m,m,b) where a=1, b=N, and m = floor((b-a)/2)
## then recursively split on the intervals (a,*,*,m) and (m,*,*,b)
box_index_rng = lambda a,b: (a, int((a+b)/2), int((a+b)/2), b)

## Statically generate all boxes spanning the integer grid [a,b] x [a,b]
## in a top-down approach
def generate_boxes(a: int, b: int, res: list = []):
  res.append((a, (a+b) // 2, (a+b) // 2, b))
  if abs(a-b) <= 1: 
    return  
  else:
    generate_boxes(a, (a+b) // 2, res)
    generate_boxes((a+b) // 2, b, res)

boxes = []
generate_boxes(0, len(K), boxes)

## All the persistence pairs! 
from more_itertools import collapse, chunked, unique_everseen
pairs = chunked(collapse([pairs_in_box(box) for box in boxes]), 2)
pairs = [tuple(p) for p in unique_everseen(pairs)]
pairs = np.array(pairs, dtype=[('birth', 'f4'), ('death', 'f4')])

dgm_coned


# %% Construct the diagram 
from pbsig.persistence import ph_rank
D = boundary_matrix(K, p=1)
ph_rank(D)







## Screw pylops, wayyyy too many inconsistencies, lets just roll our own. 
import pylops
lo_scipy = aslinearoperator(D)
lo_pylops = pylops.aslinearoperator(lo_scipy)
lo_pylops.matvec(np.random.uniform(size=11))
lo_pylops.apply_columns([0,1,2,3]).adjoint().matvec(np.random.uniform(size=4))

isinstance(lo_pylops.apply_columns([0,1,2,3]), pylops.LinearOperator)







## 
mu = mu_query(i,j,k,l)
creators = list(bisection_tree_top_down(D,mu_query,(i,j,k,l),mu,splitrows=True))

creators = list(bisection_tree_top_down(D,mu_query,(0,0,k,l),1,splitrows=False))
list(bisection_tree_top_down(D,mu_query,(1,1,k,l),splitrows=False))
list(bisection_tree_top_down(D,mu_query,(2,2,k,l),splitrows=False))

creators = []
bisection_tree(D,i,j,k,l,creators)

destroyers = []
bisection_tree2(D,2,2,k,l,destroyers)







seed = 259
G = nx.connected_watts_strogatz_graph(n=15, k=5, p=0.10, seed=seed)
X = np.array(list(nx.fruchterman_reingold_layout(G, seed=seed).values()))
fv = X @ np.array([0,1])
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
mu_query(L, R = (fi,fj,fk,fl), f=lambda s: fv[s].max() + 1.0)


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