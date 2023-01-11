import numpy as np 
from pbsig import * 
from pbsig.linalg import * 
from pbsig.datasets import mpeg7
from pbsig.betti import *
from pbsig.pht import pht_preprocess_pc, rotate_S1
from pbsig.persistence import *
from pbsig.simplicial import cycle_graph, MutableFiltration
import matplotlib.pyplot as plt 
from pbsig.vis import plot_complex

# %% Load dataset 
# NOTE: PHT preprocessing centers and scales each S to *approximately* the box [-1,1] x [-1,1]
dataset = { k : pht_preprocess_pc(S, nd=64) for k, S in mpeg7(simplify=150).items() }

# t1 = dataset[('turtle',1)]
# t2 = dataset[('turtle',2)]
# plt.plot(*t1.T); plt.plot(*t2.T)
# t1 = t1 @ np.array([[-1, 0], [0, 1]])
# plt.plot(*t1.T); plt.plot(*t2.T)
# R, t = find_rigid_alignment(t1,t2)
# plt.plot(*t1.T); plt.plot(*(t2 @ R.T).T)
# plt.plot(*(t1 @ R).T); plt.plot(*t2.T)


#%% Compute mu queries 
X = dataset[('turtle',1)]
S = cycle_graph(X)
L = up_laplacian(S, p=0, form='lo')

# fv = X @ np.array([0,1])
# K = MutableFiltration(S, f = lambda s: max(fv[s]))
# K = K.reindex_keys(np.arange(len(K)))

## Sample a couple rectangles in the upper half-plane and plot them 
R = sample_rect_halfplane(1, area=(0.10, 1.00))

## Plot all the diagrams, viridas coloring, as before
from pbsig.color import bin_color
E = np.array(list(S.faces(1)))
vir_colors = bin_color(range(64))
for ii, v in enumerate(uniform_S1(64)):
  dgm = ph0_lower_star(X @ v, E, max_death='max')
  plt.scatter(*dgm.T, color=vir_colors[ii], s=1.25)
  i,j,k,l = R[0,:]
  ij = np.logical_and(dgm[:,0] >= i, dgm[:,0] <= j) 
  kl = np.logical_and(dgm[:,1] >= k, dgm[:,1] <= l) 
  #if np.any(np.logical_and(ij, kl)): print(ii)

ax = plt.gca()
for i,j,k,l in R:  
  rec = plt.Rectangle((i,k), j-i, l-k)
  rec.set_color('#0000000f')
  ax.add_patch(rec)

from pbsig.simplicial import SimplicialComplex
from pbsig.betti import MuSignature

def directional_transform(X: ArrayLike):
  def _transform(theta: float):
    fv = X @ np.array([np.cos(theta), np.sin(theta)])
    return lambda s: max(fv[s])
  return _transform

Theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
dt = directional_transform(X)
F = [dt(theta) for theta in Theta]
sig = MuSignature(S, F, R[0,:])

import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(sig.precompute)
profile.add_function(sig.L._matvec)
profile.add_function(sig.L._matmat)
profile.enable_by_count()
sig.precompute()
profile.print_stats(output_unit=1e-3)

## compare signatures manually 
Sigs = {}
for k, X in dataset.items():
  S = cycle_graph(X)
  dt = directional_transform(X)
  F = [dt(theta) for theta in Theta]
  Sigs[k] = MuSignature(S, F, R[0,:])

## Precompute the singular values associated with the DT for each shape 
for ii, sig in enumerate(Sigs.values()):
  print(ii)
  sig.precompute(pp=0.30, tol=1e-4, w=0.30)

s1 = Sigs[('watch',1)]()
s2 = Sigs[('watch',2)]()
plt.plot(s1); plt.plot(s2)

plt.plot(*dataset[('watch',1)].T)
plt.plot(*dataset[('watch',2)].T)

## Stabilize signature with PoU 


def pointwise_dist(a,b, check_reverse: bool = True):
  d1 = np.linalg.norm(b-phase_align(a,b))
  d2 = np.linalg.norm(b-phase_align(np.flip(a),b)) if check_reverse else d1
  return min(d1, d2)

## See if they align
from pbsig.signal_tools import phase_align
import bokeh 
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import *
from bokeh.plotting import figure, curdoc
from bokeh.plotting import figure, show, curdoc

output_notebook()

p = figure(height=200, width=400)
p.outline_line_color = None
p.grid.grid_line_color = None
params = dict(smoothing = (0.10, 1.2, 0))
mu_sig = lambda k,m: Sigs[(k, m)](**params)

keys = ['turtle', 'watch', 'bone', 'bell', 'bird', 'beetle', 'dog'] # 'chicken', 'lizzard'
colors = ['firebrick', 'orange', 'black', 'blue', 'green', 'purple', 'yellow']
for t, c in zip(keys, colors):
  p.line(Theta, mu_sig(t,1), color=c)
  p.line(Theta, phase_align(mu_sig(t,2), mu_sig(t,1)), color=c)
  p.line(Theta, phase_align(mu_sig(t,3), mu_sig(t,1)), color=c)
  p.line(Theta, phase_align(mu_sig(t,4), mu_sig(t,1)), color=c)
  p.line(Theta, phase_align(mu_sig(t,5), mu_sig(t,1)), color=c)
  p.line(Theta, phase_align(mu_sig(t,6), mu_sig(t,1)), color=c)
show(p)

## Make a plot of each signal on a different horizontal slice, and then pictures 
## of the shapes on horizontal slices on the right 


for s1,s2 in product(['bone', 'watch', 'bird'], ['bone', 'watch', 'bird']):
  for i,j in combinations(range(1, 4), 2):
    print(f"{s1}-{i}, {s2}-{j} ~ d={pointwise_dist(mu_sig(s1,i), mu_sig(s2,j))}")


from pbsig.signal_tools import phase_align
from itertools import combinations
filter(lambda k: k[0] in [""], Sigs.keys())
n = len(Sigs)
d = []

for (k1,k2) in combinations(Sigs.keys(),2):
  a,b = Sigs[k1](**params), Sigs[k2](**params)
  d1 = np.linalg.norm(b-phase_align(a,b))
  d2 = np.linalg.norm(b-phase_align(np.flip(a),b))
  d.append(min(d1,d2))

from pbsig.color import bin_color
C = np.floor(bin_color(d)*255).astype(int)
D = np.zeros((n,n), dtype=np.uint32)
D_view = D.view(dtype=np.int8).reshape((n, n, 4))
for cc, (i,j) in enumerate(combinations(range(n),2)):
  D_view[i,j,0] = D_view[j,i,0] = C[cc,0]
  D_view[i,j,1] = D_view[j,i,1] = C[cc,1]
  D_view[i,j,2] = D_view[j,i,2] = C[cc,2]
  D_view[i,j,3] = D_view[j,i,3] = 255

min_col = (bin_color([0, 1])[0,:]*255).astype(int)
for i in range(n):
  D_view[i,i,0] = 68
  D_view[i,i,1] = 2
  D_view[i,i,2] = 85
  D_view[i,i,3] = 255

p = figure(width=200, height=200)
p.x_range.range_padding = p.y_range.range_padding = 0
# p.y_range.flipped = True
# p.y_range = Range1d(0,-10)
# p.x_range = Range1d(0,10)
p.image_rgba(image=[np.flipud(D)], x=0, y=0, dw=10, dh=10)
show(p)

d = np.zeros((n,n), dtype=np.uint32)
d_view = d.view(dtype=np.int8).reshape((n, n, 4))
np.floor(bin_color(D)*255).astype(int)




from scipy.spatial.distance import pdist, squareform


from pbsig.utility import * 

procrustes_dist_cc(dataset[('turtle',1)], dataset[('turtle',2)], do_reflect=True)
procrustes_dist_cc(*procrustes_cc(A, B))








## Index filter values <=> index 
f_index_set = np.array(list(K.keys()))

## Given rectangle, find its closest equivalent box via index 
i = np.searchsorted(f_index_set, R[0], side='left')
j = np.searchsorted(f_index_set, f_index_set[np.searchsorted(f_index_set, R[1])], side='right')
k = np.searchsorted(f_index_set, R[2], side='left')
l = np.searchsorted(f_index_set, f_index_set[np.searchsorted(f_index_set, R[3])], side='right')


## Create a signature via mu queries
r = R[0,:]
sig = []
for v in uniform_S1(132):
  fv = X @ np.array(v)
  sig.append(mu_query(L, r, f=lambda s: max(fv[s]), w=0.20))

plt.plot(sig)

dgm = barcodes(K)
plt.scatter(dgm['birth'], dgm['death'])

fv = X @ np.array([0,1])
mu_query(L, R, f=lambda s: max(fv[s]), w=1.15)

r = np.array([-np.inf, -0.75456026, -0.29686413, np.inf])
mu_query(L, r, f=lambda s: max(fv[s]), w=0.0)



import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(smooth_rank)
profile.add_function(trace_threshold)
profile.add_function(L._matvec)
profile.enable_by_count()
smooth_rank(L)
profile.print_stats(output_unit=1e-3)

mu_query(L, i, j)

# %% Generate 
X = dataset[('turtle',1)]

L = up_laplacian(S, p=0, form='array')
sum(np.sort(primme.eigsh(L, k=L.shape[0], ncv=L.shape[0], return_eigenvectors=False)))



L = up_laplacian(S, p=0, form='lo')
ew_lap = np.sort(primme.eigsh(L, k=L.shape[0], ncv=L.shape[0], return_eigenvectors=False))
np.allclose(ew_lap, ew_cf) # True

# ph0_lower_star(fv: ArrayLike, E: ArrayLike)
fv = X @ np.array([0,1])
K = MutableFiltration(S, f = lambda s: max(fv[s]))

barcodes(K)
boundary_matrix(K)

ph(K)
lower_star_pb


import numpy as np 
from scipy.linalg import toeplitz
from scipy.sparse import csc_array, diags
m = 50
T = csc_array(toeplitz([2, -1] + [0]*(m-3) + [-1]))

ew_np = np.sort(np.linalg.eigvalsh(T.todense()))
ew_cf = np.sort([2.0-2.0*np.cos(2*np.pi*k/m) for k in range(m)])
np.allclose(ew_np, ew_cf) # True

import primme
ew_pm = np.sort(primme.eigsh(T, k=m, ncv=m, return_eigenvectors=False))
np.allclose(ew_pm, ew_cf) # True

import primme
np.random.seed(1234)
m = 150
T = csc_array(toeplitz([2, -1] + [0]*(m-3) + [-1]))
ew_pm = np.sort(primme.eigsh(T, k=m, ncv=m, maxiter=5000, return_eigenvectors=False))
np.allclose(ew_pm, ew_cf) 

np.random.seed(1234)
m = 100
T = csc_array(toeplitz([2, -1] + [0]*(m-3) + [-1]))
w = np.random.uniform(size=m, low=0, high=1)
#w[np.random.choice(np.arange(m), size=5)] = 0
WT = diags(w) @ T @ diags(w)
# plt.spy(WT)
ew_pm = np.sort(primme.eigsh(WT, k=m, return_eigenvectors=False))
ew_np = np.sort(np.linalg.eigvalsh(WT.todense()))
np.allclose(ew_pm, ew_np) 
any(ew > WT.trace())

np.allclose(ew_pm, ew_cf) # True

# L = up_laplacian(S, p=0, form='array')
# ew_lap = np.sort(primme.eigsh(L, k=L.shape[0], ncv=L.shape[0], return_eigenvectors=False))
# np.allclose(ew_lap, ew_cf) # True


plot_complex(S, pos = X)
smooth_rank(L)


## Unable to replicate non-convergence behavior of integer laplacian 
import primme
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
n = 10
M = np.random.uniform(size=(n,n), low=-1.0, high=1.0)
Q,R = np.linalg.qr(M)
d = np.round(np.random.uniform(size=n, low=0.0, high=10.0))
d[np.random.choice(range(n), size=5)] = 0
A = Q @ diags(d) @ Q.T
eigsh(A, k=4, return_eigenvectors=False, which='LM')
primme.eigsh(A, k=n, return_eigenvectors=False, which='LM')
primme.eigsh(A, k=n, ncv=n, return_eigenvectors=False, which='LM')


from scipy.linalg import * 
c,r = np.random.uniform(size=5), np.random.uniform(size=5)
P = np.fliplr(np.eye(5))

x = np.random.uniform(size=5, low=-1, high=1)
T = toeplitz(c, r)
b = T @ x 
solve_toeplitz((c,r), b) # works only if every principle subminor is full rank

H = hankel(np.flip(c), r)
np.allclose(T - P @ H, 0)
np.allclose(H @ x, P @ b)
np.allclose(P @ (H @ x), b)


toeplitz([2,-1,], )


## Find proportion of eigenvalues needed on average to compute 90% of spectrum
# n = 150
# mu = 0
# for i in range(1500):
#   Q,R = np.linalg.qr(np.random.uniform(size=(n,n), low=-1.0, high=1.0))
#   ew = np.linalg.eigvalsh(Q @ diags(np.random.uniform(size=n, low=0, high=1.0)) @ Q.T)
#   cdf = np.cumsum(np.sort(ew)[::-1])/sum(ew)
#   # plt.plot()
#   mu += max(np.flatnonzero(cdf <= 0.90))/len(cdf)
# mu / 1500

from scipy.sparse.linalg import eigsh
eigsh(L, k=148, tol=1e-6, return_eigenvectors=False)
primme.eigsh(A, k=149, ncv=150, tol=1e-6, maxiter=A.shape[0]*50, return_eigenvectors=False, return_unconverged=True, which='LM')


D1 = boundary_matrix(K, p=1).tocsc()

E = K['edges']
S = np.sign(D1.data)
nev = 120
# fv = X @ np.array([1,0])
# D1.data = S * np.repeat(fv[E].max(axis=1), 2)
# LE = D1 @ D1.T
# nev = np.flatnonzero(np.cumsum(LE.diagonal()) > 0.90*sum(LE.diagonal()))[0]

SS = np.zeros(shape=(nev, 132))
for j, fv in enumerate(rotate_S1(X, nd=132, include_direction=False)):
  D1.data = S * np.repeat(fv[E].max(axis=1), 2)
  LE = D1 @ D1.T
  SS[:,j] = np.sqrt(primme.eigsh(LE, k=nev, which='LM', tol=1e-6, return_eigenvectors=False))

## Signature across the rotation
# plt.plot(SS.sum(axis=0))
# plt.plot((SS**2).sum(axis=0))
plt.plot(((SS**2)/(SS**2 + 1.2)).sum(axis=0))
ax = plt.gca()
ax.set_ylim(0, 120)



# %% 
dataset[('turtle',1)].mean(axis=0)

# V = [v for fv, v in rotate_S1(X_shape, 32)]
# V = np.array([v.flatten() for v in V])

# u = shape_center(X_shape, "hull")
# K = 256
# Lambdas0 = [min(fv) for fv, v in rotate_S1(X_shape, K)]
# c = (1/K)*sum([l*v for v,l in zip(V, Lambdas0)])
# print(c)
# X_shape = X_shape - c.flatten() 

X_shape = dataset[('turtle',1)]
V = np.array(list(uniform_S1(n=128)))
u1 = shape_center(X_shape, "directions", V = V)  ## barycenter of set of direction vectors
u2 = shape_center(X_shape, "polygon")            ## barycenter of polygon
u3 = shape_center(X_shape, "barycenter")         ## barycenter of points
u4 = shape_center(X_shape, "hull")               ## barycenter of convex hull 
u5 = shape_center(X_shape, "bbox")               ## barycenter of bounding box

## Trying out ideas 
# edge_bc = np.array([0.5*u + 0.5*v for u,v in cycle_window(X_shape)])
# areas = [np.linalg.norm(u-v) for u,v in cycle_window(X_shape)]
# U = np.array([bc*a for bc, a in zip(edge_bc, areas)])
# U.sum(axis=0)/sum(areas)
# U.sum(axis=0)/Polygon(X_shape).area
# shape_center(X_shape, "directions", V = U)

## Show different means 
plt.plot(*X_shape.T)
plt.scatter(*np.vstack((u1,u2,u3,u4,u5)).T,s=15.5, c=['red','yellow','green','blue','orange'])

# A = X_shape - u1
# plt.hist([min(X_shape @ v) for v in V])
# L = -sum([min(A @ v) for v in V])
# B = (V.shape[0]/L)*A
# sum([min(B @ v) for v in V])
# plt.plot(*B.T)



## Show the shapes 
from itertools import *
valid_keys = list(filter(lambda k: k[0] in shape_types, dataset.keys()))
fig, axs = plt.subplots(len(shape_types),len(shape_nums),figsize=(16,6), dpi=220)
for i, class_type in enumerate(shape_types):
  for j, k in enumerate(filter(lambda k: k[0] == class_type, valid_keys)):
    S = dataset[k]
    S_closed = np.array(list(islice(cycle(S), S.shape[0]+1)))
    axs[i,j].plot(*S.T, linewidth=0.50)
    axs[i,j].axis('off')
    axs[i,j].set_aspect('equal')

## Show the shapes super-imposed
valid_keys = list(filter(lambda k: k[0] in shape_types, dataset.keys()))
fig = plt.figure(figsize=(2,2), dpi=220)
ax = fig.gca()
for i, class_type in enumerate(shape_types):
  for j, k in enumerate(filter(lambda k: k[0] == class_type, valid_keys)):
    S = dataset[k]
    S_closed = np.array(list(islice(cycle(S), S.shape[0]+1)))
    ax.plot(*S_closed.T, linewidth=0.50)
    ax.axis('off')
    ax.set_aspect('equal')

## Procrustes should be invariant to preprocessing, rotation, and matching
A, B = dataset[('turtle',1)], dataset[('turtle',2)]
procrustes_dist_cc(A, B) == procrustes_dist_cc(*procrustes_cc(A, B))

## Manual preprocessing
# from pbsig.utility import procrustes_cc
# pairs = [[('turtle',1), ('turtle',2)], [('watch',1), ('watch',2)], [('chicken',1),('chicken',2)]]
# for p1, p2 in pairs: 
#   dataset[p1], dataset[p2] = procrustes_cc(dataset[p1], dataset[p2], matched=False, preprocess=False, do_reflect=True)

# plt.plot(*dataset[('turtle',1)].T);plt.plot(*dataset[('turtle',2)].T)
# plt.plot(*dataset[('chicken',1)].T);plt.plot(*dataset[('chicken',2)].T)
# plt.plot(*dataset[('watch',1)].T);plt.plot(*dataset[('watch',2)].T)

from pbsig.pht import pht_0
dataset = mpeg7(simplify=138)
X = dataset[('turtle', 1)]
E = np.array(list(cycle_window(range(X.shape[0]))))
X_dgms = pht_0(X,E,nd=132,transform=False,replace_inf=True)
X_dgms_1 = [ph0_lower_star(f, E, max_death="max") for f in rotate_S1(X, 132, include_direction=False)]
plot_fake_vines(X_dgms)
plot_fake_vines(X_dgms_1)

lex_sort = lambda X: X[np.lexsort(np.rot90(X)),:]
for ii, (A, B) in enumerate(zip(X_dgms, X_dgms_1)):
  A, B = lex_sort(A).astype(np.float32), lex_sort(B).astype(np.float32)
  assert len(A) == len(B)
  assert max(abs(A - B).flatten()) <= np.finfo(np.float32).eps

F = list(rotate_S1(X, 132, include_direction=False))
f = F[ii]
lower_star_ph_dionysus(f, E=E, T=[])[0].shape
ph0_lower_star(f, E, collapse=True).shape

plt.scatter(*X.T, c=F[ii])
for k,(x,y) in enumerate(X): plt.text(x,y,s=str(k))

f = F[ii]
d1 = lower_star_ph_dionysus(f, E=E, T=[])[0]
[np.argmin(abs(f - b)) for b in d1[:,0]]
[np.argmin(abs(f - d)) for d in d1[:,1] if d != np.inf]

fv = f
d2 = ph0_lower_star(F[ii], E, collapse=True)
[np.argmin(abs(f - b)) for b in d2[:,0]]
[np.argmin(abs(f - d)) for d in d2[:,1] if d != np.inf]

## Compute PHT
from pbsig.pht import pht_0_dist, pht_preprocess_pc, pht_0
X = [dataset[k] for k in dataset.keys()]
E = [np.array(list(zip(range(x.shape[0]), np.roll(range(x.shape[0]), -1)))) for x in X]
dgms = [pht_0(x,e,nd=132,transform=False,replace_inf=True) for x,e in zip(X,E)]

## Resolve infimum by W1 distance
D = pht_0_dist(dgms, preprocess=False, progress=True, diagrams=True)

## Don't resolve infimum rotation: assumes manual rotation has been done
D = pht_0_dist(dgms, mod_rotation=False, preprocess=False, progress=True, diagrams=True)

## Is the PHT a discriminator?
assert len(np.unique(D)) == len(D)

## Show PHT distance matrix
# from pbsig.color import scale_interval
# scaling = 'linear' # linear, equalize, logarithmic
# plt.imshow(squareform(scale_interval(D, scaling=scaling)))
# plt.suptitle(f"MPEG dataset - PHT")
# plt.title(f"{scaling} scaling")

## Show PHT distance matrix better
from pbsig.color import scale_interval, heatmap, annotate_heatmap
D_im = squareform(scale_interval(D/50000, scaling='linear'))
#I = np.array([0,1,2,3,4,5,6,7,8,9,17,18,19], dtype=int)
I = np.fromiter(range(D_im.shape[0]), dtype=int)
fig, ax = plt.subplots()
labels = np.array([name+str(i) for name,i in dataset.keys()])
im, cbar = heatmap(D_im[np.ix_(I,I)], labels[I], labels[I], ax=ax, cmap="YlGn", cbarlabel="PHT distance")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
fig.tight_layout()
plt.show()

## Show procrustes distance between closed curves 
from pbsig.utility import procrustes_dist_cc
from itertools import combinations
pd_curves = np.array([procrustes_dist_cc(A, B, matched=False, preprocess=False) for A, B in combinations(dataset.values(), 2)])

fig, ax = plt.subplots()
labels = np.array([name+str(i) for name,i in dataset.keys()])
im, cbar = heatmap(squareform(pd_curves), labels, labels, ax=ax, cmap="YlGn", cbarlabel="Procrustes distance")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
fig.tight_layout()
plt.show()

## Why are the two chickens and turtles far from each other?
dataset.keys() # d(turtle1, turtle2) > d(turtle1, watch1)?

T1, T2, T3 = dataset[('turtle',1)], dataset[('turtle',2)], dataset[('watch',1)]
# R_refl = np.array([[-1, 0], [0, 1]])
# dataset[('turtle',1)] = T1 = T1 @ R_refl
plt.plot(*T1.T);plt.scatter(*T1.T, c=T1 @ np.array([0,1]))
plt.plot(*T2.T);plt.scatter(*T2.T, c=T2 @ np.array([0,1]))
plt.plot(*T3.T);plt.scatter(*T3.T, c=T3 @ np.array([0,1]))

E = np.array(list(cycle_window(range(T1.shape[0]))))
T1_dgms = pht_0(T1,E,nd=132,transform=False,replace_inf=True)
T2_dgms = pht_0(T2,E,nd=132,transform=False,replace_inf=True)
T3_dgms = pht_0(T3,E,nd=132,transform=False,replace_inf=True)

## Assume shapes are aligned; look at the W-2 distance over S1 
dw_12 = np.array([wasserstein_distance(d1,d2) for d1, d2 in zip(T1_dgms, T2_dgms)])
dw_13 = np.array([wasserstein_distance(d1,d2) for d1, d2 in zip(T1_dgms, T3_dgms)])
dw_23 = np.array([wasserstein_distance(d1,d2) for d1, d2 in zip(T2_dgms, T3_dgms)])

plt.plot(dw_12,c='red');plt.plot(dw_13,c='blue'); plt.plot(dw_23,c='orange')
## red is turtle1, turtle2
## blue is turtle1, watch1
## orange is turtle2, watch1

plot_fake_vines(T1_dgms);plot_fake_vines(T2_dgms);plot_fake_vines(T3_dgms)


## Shells example
from pbsig.shape import shells
S = [shells(A, k=30) for A in dataset.values()]
dS = np.array([np.linalg.norm(s1-s2) for s1, s2 in combinations(S, 2)])

fig, ax = plt.subplots()
labels = np.array([name+str(i) for name,i in dataset.keys()])
im, cbar = heatmap(squareform(dS), labels, labels, ax=ax, cmap="YlGn", cbarlabel="Shells distance")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
fig.tight_layout()
plt.show()

## PB curve example
from pbsig.betti import lower_star_betti_sig
F = list(rotate_S1(dataset[('turtle',1)], nd=132, include_direction=False))
T1_dgms = [lower_star_ph_dionysus(f, E, [])[0] for f in F]
for i, f in zip(range(len(T1_dgms)), F):
  dgm = T1_dgms[i]
  dgm[dgm[:,1] == np.inf,1] = max(f)
  T1_dgms[i] = dgm

R = [[-0.25, 0.05], [-1.2,0.05], [-0.25,1.15], [-1.2,1.15]]
E = np.array(list(cycle_window(range(len(F[0])))))
B = [lower_star_betti_sig(F, E, nv=len(F[0]), a=a, b=b, method="rank") for a,b in R]

## 
# plot_dgm(lower_star_ph_dionysus(F[0], E, [])[0])

fig, ax = plot_fake_vines(T1_dgms)
ax.plot(*(np.array(R)[[0,1,3,2,0]]).T, linewidth=0.20)

B_truth = [np.array([sum(np.logical_and(d[:,0] <= a, d[:,1] > b)) for d in T1_dgms]) for a,b in R]
B_truth[0] == B[0]
B_truth[1] == B[1]
B_truth[2] == B[2] # B[2] is wrong?
B_truth[3] == B[3]
# B[0] - B[1] - B[2] + B[3] 
plt.plot(B[0] - B[1] - B[2] + B[3])

i = min(np.flatnonzero(B_truth[2] != B[2].astype(int)))
a,b = R[2]
b0_summands = lower_star_betti_sig([F[i]], E, nv=len(F[i]), a=a, b=b, method="rank", keep_terms=True).astype(int) 

sum(F[i] <= a) == b0_summands[0,0]
fig, ax = plot_dgm(T1_dgms[i])
ax.plot(*(np.array(R)[[0,1,3,2,0]]).T, linewidth=0.20)

lower_star_betti_sig([F[0]], E, nv=len(F[0]), a=a, b=b, method="rank", keep_terms=True)

K = { 'vertices' : np.fromiter(range(len(F[i])), dtype=int), 'edges': E }
V_names = [str(v) for v in K['vertices']]
E_names = [str(tuple(e)) for e in K['edges']]
f0, e0 = F[i], F[i][E].max(axis=1)
VF0 = dict(zip(V_names, f0))
EF0 = dict(zip(E_names, e0))
D0, D1 = boundary_matrix(K, p=(0,1))
D1 = D1[np.ix_(np.argsort(f0), np.argsort(e0))]
R0, R1, V0, V1 = reduction_pHcol(D0, D1)
ii = sum(np.array(sorted(f0)) <= a)
jj = sum(np.array(sorted(e0)) <= b)
persistent_betti(D0, D1, ii, jj, summands=True)
np.linalg.matrix_rank(D1[ii:,:(jj+1)].A)

D1_true = D1.copy()

## Test case: 
from itertools import product
fv, ev = F[0], F[0][E].max(axis=1)

K = { 'vertices' : np.fromiter(range(len(fv)), dtype=int), 'edges': E }
V_names = [str(v) for v in K['vertices']]
E_names = [str(tuple(e)) for e in K['edges']]
VF0 = dict(zip(V_names, fv))
EF0 = dict(zip(E_names, ev))
D0, D1 = boundary_matrix(K, p=(0,1))
D1 = D1[np.ix_(np.argsort(fv), np.argsort(ev))]
for i,j in product(fv, ev):
  if i < j:
    ## This requires D1 to be in sorted order, based on y
    ii, jj = np.searchsorted(fv, i), np.searchsorted(ev,j)
    p2 = np.array(persistent_betti(D0, D1, i=ii, j=jj, summands=True))

    ## This does not 
    nv = len(fv)
    p1 = lower_star_betti_sig([fv], p_simplices=E, nv=nv, a=i,b=j,method="rank", keep_terms=True).flatten().astype(int)
    if p1[0] == 0 and p2[0] == 0:
      continue
    assert all(abs(p1) == abs(p2)), "Failed assertion"

np.array(list(zip(*W.nonzero())))

## Compare 
W_full = np.zeros(shape=(D1.shape[0], D1.shape[0]))
W = D1[ii:,:(jj+1)].tocsc().copy()
W @ W.T
L # \# of non-zeros

## betti 
T = D1[np.ix_(np.argsort(f0), np.argsort(e0))]
T.sort_indices()
T.nonzero()[0]

BW = T[ii:,:(jj+1)]

np.linalg.matrix_rank(T[ii:,:(jj+1)].A)





d = lower_star_ph_dionysus(F[i], E, [])[0]
sum(np.logical_and(d[:,0] <= a, d[:,1] > b))


F1 = list(rotate_S1(dataset[('turtle',1)], nd=132, include_direction=False))
F2 = list(rotate_S1(dataset[('turtle',2)], nd=132, include_direction=False))
F3 = list(rotate_S1(dataset[('watch',1)], nd=132, include_direction=False))
B1 = [lower_star_betti_sig(F1, E, nv=len(F1[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]
B2 = [lower_star_betti_sig(F2, E, nv=len(F2[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]
B3 = [lower_star_betti_sig(F3, E, nv=len(F3[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]
#a,b = -0.25, 1.15
# [[-0.25, 0.05], [-1.2, 0.05], [-0.25, 1.15], [-1.2, 1.15]]

S1 = B1[0] - B1[1] - B1[2] + B1[3]
S2 = B2[0] - B2[1] - B2[2] + B2[3]
S3 = B3[0] - B3[1] - B3[2] + B3[3]
fig, axs = plt.subplots(1,2,figsize=(8,2), dpi=150)
axs[0].plot(S1, color='red', label='Turtle 1')
axs[0].plot(S2, color='blue', label='Turtle 2')
axs[0].plot(S3, color='yellow', label='Watch 1')
axs[1].plot(abs(S1-S2), color='purple', label='|T1-T2|')
axs[1].plot(abs(S1-S3), color='orange', label='|T1-W1|')
axs[1].plot(abs(S2-S3), color='green', label='|T2-W1|')
axs[0].legend()
axs[1].legend()




# cm.get_cmap('viridis') # purple == low, yellow == high





## Check reversed, do optimal rotation, and then optimal sliding 
from pbsig.utility import shift_curve, winding_distance
from pyflubber.closed import prepare
T1_r = np.array(list(reversed(T1)))

## Original 
A,B = T1,T2
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')

## Shifted 
T1_p, T2_p = prepare(T1, T2)
A,B = T1_p,T2_p
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')

A = shift_curve(A, B, lambda X,Y: np.linalg.norm(X-Y, 'fro'))
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')

## manual rolling
# from procrustes import rotational
# pro_error = lambda X,Y: rotational(X, Y, pad=False, translate=False, scale=False).error
pro_error = lambda X,Y: np.linalg.norm(X @ Kabsch(X, Y) - Y, 'fro')
os = np.argmin([pro_error(T1_p, np.roll(T2_p, offset, axis=0)) for offset in range(T1_p.shape[0])])
A,B = T1_p, np.roll(T2_p, os, axis=0)
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')

## Then rotated?
R = Kabsch(A, B)
AR,BR = A @ R, B @ R
plt.plot(*AR.T);plt.plot(*BR.T)
plt.scatter(*AR.T, c=range(AR.shape[0]));plt.scatter(*BR.T, c=range(BR.shape[0]))
plt.gca().set_aspect('equal')

## Replace and check!
dataset[('turtle',1)] = A
dataset[('turtle',2)] = B

# res = rotational(T1, T2, pad=False, translate=False, scale=False)
# plt.plot(*(res.new_a@res.t).T)
# plt.plot(*res.new_b.T)
# plt.gca().set_aspect("equal")



from procrustes import orthogonal
from procrustes.rotational import rotational
res = rotational(T1, T2, pad=False, translate=False, scale=False)
plt.plot(*(res.new_a@res.t).T)
plt.plot(*res.new_b.T)
plt.gca().set_aspect("equal")
res.error

## Surprising!
np.linalg.norm((res.new_a@res.t) - res.new_b, "fro")**2

## Manually verify 
rot_mat = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
opt_theta = Theta[np.argmin([np.linalg.norm((T1@rot_mat(t))-T2, 'fro') for t in Theta])]
plt.plot(*(T1@rot_mat(opt_theta)).T)
plt.plot(*T2.T)
plt.gca().set_aspect("equal")
np.linalg.norm(T1@rot_mat(opt_theta) - T2, "fro")**2

plt.plot(*(T1@res.t).T)
plt.plot(*T2.T)
plt.gca().set_aspect("equal")
np.linalg.norm(T1@res.t - T2, "fro")**2

## Need correct correspondences for Procrustes! 
from pyflubber.closed import prepare
T1_p, T2_p = prepare(T1, T2)
plt.plot(*T1_p.T)
plt.plot(*T2_p.T)

res = rotational(T1_p, T2_p, pad=False, translate=False, scale=False)
plt.plot(*(res.new_a@res.t).T)
plt.plot(*res.new_b.T)
plt.gca().set_aspect("equal")
res.error

from pyflubber.closed import prepare
T1_p, T2_p = prepare(T1, T2)


res = rotational(T1_p, T2_p, pad=False, translate=False, scale=False)
T1_p, T2_p = res.new_a@res.t, res.new_b
plt.plot(*T1_p.T)
plt.plot(*T2_p.T)
plt.gca().set_aspect("equal")

plt.plot(*shift_curve(T1, T2).T)
plt.plot(*T2.T)
plt.gca().set_aspect('equal')


plt.plot(*shift_curve(T1, T2, objective=pro_error).T)
plt.plot(*T2.T)
plt.gca().set_aspect('equal')

os = np.argmax([objective(reference, np.roll(line, offset, axis=0)) for offset in range(len(line))])
plt.plot(*(np.roll(T1,os,axis=0)).T)
plt.plot(*T2.T)

T1_p = np.roll(T1,os,axis=0)
# procrustes_error(T1, T2)
# procrustes_error(T1_p, T2)
res = rotational(T1_p, T2, pad=False, translate=False, scale=False)
T1_n, T2_n = res.new_a@res.t, res.new_b
plt.plot(*T1_n.T)
plt.plot(*T2_n.T)

rot_mat = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
opt_theta = Theta[np.argmin([np.linalg.norm((T1_n@rot_mat(t))-T2_n, 'fro')**2 for t in Theta])]
plt.plot(*(T1_n@rot_mat(opt_theta)).T)
plt.plot(*T2_n.T)
plt.gca().set_aspect("equal")
np.linalg.norm(T1@rot_mat(opt_theta) - T2, "fro")**2

A_manual = T1_n@rot_mat(np.pi + np.pi/12)
plt.plot(*A_manual.T)
plt.scatter(*A_manual.T, c=range(A_manual.shape[0]))

plt.plot(*T2_n.T)
plt.scatter(*T2_n.T, c=range(T2_n.shape[0]))


np.linalg.norm((T1_n@rot_mat(np.pi + np.pi/12)) - T2_n, 'fro')
np.linalg.norm(() - T2_n, 'fro')

A_opt = T1_n@rot_mat(opt_theta)
plt.plot(*A_opt.T)
plt.scatter(*A_opt.T, c=range(A_opt.shape[0]))
plt.plot(*T2_n.T)
plt.scatter(*T2_n.T, c=range(T2_n.shape[0]))



# from scipy.linalg import orthogonal_procrustes
# R, scale = orthogonal_procrustes(T1, T2)
# plt.plot(*(T1 @ R).T)
# plt.plot(*(T2).T)

pht_0_dist([T1_dgms, T2_dgms], preprocess=False, progress=True, diagrams=True)

plot_fake_vines(T1_dgms)
plot_fake_vines(T2_dgms)



def plot_fake_vines(dgms):
  from pbsig.color import bin_color
  fig, ax = plot_dgm(dgms[0])
  C = bin_color(range(len(dgms)))
  for i, dgm in enumerate(dgms):
    ax.scatter(*dgm.T, s=0.05, marker=".", color=C[i,:])
  min_x = min([min(dgm[:,0]) for dgm in dgms])
  max_x = max([max(dgm[1:,1]) for dgm in dgms])
  diff = abs(max_x-min_x)
  ax.set_xlim(min_x-0.10*diff, max_x+0.10*diff)
  ax.set_ylim(min_x-0.10*diff, max_x+0.10*diff)
  ax.set_aspect('equal')
  return(fig, ax)

winding_distance(X, Y)

from pbsig.pht import pbt_0_dist, pbt_birth_death
info = pbt_birth_death(X, E, nd=32, preprocess=True)
a,b = info[0,0]*1.50, info[1,1]/2
B = pbt_0_dist(X, E, nd=32, preprocess=True, progress=True, a=a, b=b)

## Show PBT distance matrix
plt.imshow(squareform(B))

len(np.unique(B))

import numpy as np
V1 = np.array([[0,1],[0,2],[1,1],[2,1],[3,0],[3,1],[3,2]])
V2 = np.array([[0,0],[0,1],[0,2],[1,1],[2,1],[3,0],[3,2]])
E1 = np.array([[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[3,6]])
E2 = np.array([[0,3],[1,2],[1,3],[2,3],[3,4],[4,5],[4,6]])

import networkx as nx
G1, G2 = nx.Graph(), nx.Graph()
G1.add_nodes_from(range(7))
G2.add_nodes_from(range(7))
G1.add_edges_from(E1)
G2.add_edges_from(E2)


nx.draw(G1, pos = V1)
nx.draw(G2, pos = V2)

I1 = nx.incidence_matrix(G1)
I2 = nx.incidence_matrix(G2)
#ev1 = np.linalg.svd(I1.A)[1]
#ev2 = np.linalg.svd(I2.A)[1]

I1, I2 = I1.tocsc(), I2.tocsc()
I1.sort_indices()
I2.sort_indices()
I1.data = np.tile([1,-1], int(len(I1.data)/2))
I2.data = np.tile([1,-1], int(len(I2.data)/2))

np.linalg.eigvalsh((I1 @ I1.T).A)
np.linalg.eigvalsh((I2 @ I2.T).A)



## Mininim winding distance
from itertools import combinations
from pbsig.utility import winding_distance
d = np.array([winding_distance(X, Y) for X, Y in combinations(dataset.values(), 2)])
d = (d-min(d))/(max(d)-min(d))
D = squareform(d)

fig, ax = plt.subplots()
labels = np.array([name+str(i) for name,i in dataset.keys()])
im, cbar = heatmap(D, labels, labels, ax=ax, cmap="YlGn", cbarlabel="PHT distance")
texts = annotate_heatmap(im, valfmt="{x:.1f}")
fig.tight_layout()
plt.show()


## Show the PHT transformed shapes all on top of each other, draw the lines connecting them via winding; ensure preprocessing is correct
## TODO: show VINEYARDS plot with colors 
## TODO: see if winding distance and vineyards 1-Wasserstein distance compare for e.g. convex shapes (generate random shapes?)
## how is dW different for non-convex shapes?
import ot 
from gudhi.wasserstein import wasserstein_distance
wasserstein_distance(np.array([[1.0, 1.1]]), np.array([[0.90, 1.2]]), order=2.0, internal_p=2.0, keep_essential_parts=True)



plt.imshow(squareform(d))


X1, X2 = dataset[('turtle', 1)], dataset[('turtle', 2)]


from pbsig.color import bin_color
plt.plot(*X[0].T)
plt.scatter(*X[0].T, c=bin_color(range(X[0].shape[0])))

plt.plot(*A.T)
plt.scatter(*A.T, c=bin_color(range(A.shape[0])))

plt.plot(*B.T)
plt.scatter(*B.T, c=bin_color(range(A.shape[0])))

np.linalg.norm(A - B, 'fro')

# F = interpolate(X[0], X[1], t = 1.0, closed=True)


## Testing MPEG-7 again 

dataset = mpeg7(simplify=150)
for k, S in dataset.items():
  dataset[k] = pht_preprocess_pc(S, nd=132)

R = [[-0.25, 0.05], [-1.2, 0.05], [-0.25, 1.15], [-1.2, 1.15]]
F1 = list(rotate_S1(dataset[('turtle',1)], nd=132, include_direction=False))
F2 = list(rotate_S1(dataset[('turtle',2)], nd=132, include_direction=False))
F3 = list(rotate_S1(dataset[('watch',1)], nd=132, include_direction=False))
B1 = [lower_star_betti_sig(F1, E, nv=len(F1[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]
B2 = [lower_star_betti_sig(F2, E, nv=len(F2[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]
B3 = [lower_star_betti_sig(F3, E, nv=len(F3[0]), a=a, b=b, method="nuclear", w=0.15) for a,b in R]

S1 = B1[0] - B1[1] - B1[2] + B1[3]
S2 = B2[0] - B2[1] - B2[2] + B2[3]
S3 = B3[0] - B3[1] - B3[2] + B3[3]
fig, axs = plt.subplots(1,2,figsize=(8,2), dpi=150)
axs[0].plot(S1, color='red', label='Turtle 1')
axs[0].plot(S2, color='blue', label='Turtle 2')
axs[0].plot(S3, color='yellow', label='Watch 1')
axs[1].plot(abs(S1-S2), color='purple', label='|T1-T2|')
axs[1].plot(abs(S1-S3), color='orange', label='|T1-W1|')
axs[1].plot(abs(S2-S3), color='green', label='|T2-W1|')
axs[0].legend()
axs[1].legend()






# base_dir = "/Users/mpiekenbrock/Downloads/original"
# shape_types = ["turtle", "watch", "chicken"] # "bird", "lizzard", "bone", "bell"
# shape_nums = [1,2] #3,4,5

# normalize = lambda X: (X - np.min(X))/(np.max(X)-np.min(X))*255

# def largest_contour(img: ArrayLike, threshold: int = 180):
#   _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#   contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#   contours = (contours[np.argmax([len(c) for c in contours])])
#   S = contours[:,0,:]
#   return(S)

# from pbsig.pht import pht_preprocess_pc
# from pbsig.utility import simplify_outline
# dataset = {}
# for st in shape_types:
#   for sn in shape_nums:
#     img_dir = base_dir + f"/{st}-{sn}.gif"
#     assert exists(img_dir), "Image not found"
#     im = Image.open(img_dir)
#     img_gray = normalize(np.array(im)).astype(np.uint8)
#     S = largest_contour(img_gray)
#     S = largest_contour(255-img_gray) if len(S) <= 4 else S # recompute negative if bbox was found
#     assert len(S) > 4
#     S_path = simplify_outline(S, 150)
#     X_shape = np.array([l.start for l in S_path])
#     X_shape = pht_preprocess_pc(X_shape, nd=64) # defined w.r.t number of directions! 
#     dataset[(st, sn)] = X_shape
# %%
