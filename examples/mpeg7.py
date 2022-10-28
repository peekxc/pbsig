import numpy as np 
from pbsig import * 
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt 
from os.path import exists
from PIL import Image

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

dataset = mpeg7(simplify=150)
for k, S in dataset.items():
  dataset[k] = pht_preprocess_pc(S, nd=64)

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
X = [dataset[k] for k in valid_keys]
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
from pbsig.color import heatmap, annotate_heatmap
D_im = squareform(scale_interval(D, scaling='linear'))
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
F = list(rotate_S1(dataset[('turtle',1)], n=132, include_direction=False))
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


F1 = list(rotate_S1(dataset[('turtle',1)], n=132, include_direction=False))
F2 = list(rotate_S1(dataset[('turtle',2)], n=132, include_direction=False))
F3 = list(rotate_S1(dataset[('watch',1)], n=132, include_direction=False))
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

## manual rolling
pro_error = lambda X,Y: rotational(X, Y, pad=False, translate=False, scale=False).error
os = np.argmin([pro_error(T1_p, np.roll(T2_p, offset, axis=0)) for offset in range(T1_p.shape[0])])
A,B = T1_p, np.roll(T2_p, os, axis=0)
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')

## Then rotated?
res = rotational(A, B, pad=False, translate=False, scale=False)
A,B = res.new_a @ res.t, res.new_b
plt.plot(*A.T);plt.plot(*B.T)
plt.scatter(*A.T, c=range(A.shape[0]));plt.scatter(*B.T, c=range(B.shape[0]))
plt.gca().set_aspect('equal')


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

plt.plot(*shift_curve(T1, T2, objective=procrustes_error).T)
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
