import numpy as np 
from pbsig import * 
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt 
from os.path import exists
from PIL import Image

base_dir = "/Users/mpiekenbrock/Downloads/original"
shape_types = ["turtle", "watch", "chicken"] # "bird", "lizzard", "bone", "bell"
shape_nums = [1,2] #3,4,5

normalize = lambda X: (X - np.min(X))/(np.max(X)-np.min(X))*255

def largest_contour(img: ArrayLike, threshold: int = 180):
  _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = (contours[np.argmax([len(c) for c in contours])])
  S = contours[:,0,:]
  return(S)

from pbsig.pht import pht_preprocess_pc
from pbsig.utility import simplify_outline
dataset = {}
for st in shape_types:
  for sn in shape_nums:
    img_dir = base_dir + f"/{st}-{sn}.gif"
    assert exists(img_dir), "Image not found"
    im = Image.open(img_dir)
    img_gray = normalize(np.array(im)).astype(np.uint8)
    S = largest_contour(img_gray)
    S = largest_contour(255-img_gray) if len(S) <= 4 else S # recompute negative if bbox was found
    assert len(S) > 4
    S_path = simplify_outline(S, 150)
    X_shape = np.array([l.start for l in S_path])
    X_shape = pht_preprocess_pc(X_shape, transform=True, method="barycenter")
    dataset[(st, sn)] = X_shape

V = [v for fv, v in rotate_S1(X_shape, 32)]
V = np.array([v.flatten() for v in V])

## Check V is rotationally symmetric
np.allclose([min(np.linalg.norm(V + v, axis=1)) for v in V], 0.0, atol=1e-14)

K = 256
Lambdas0 = [min(fv) for fv, v in rotate_S1(X_shape, K)]
c = (1/K)*sum([l*v for v,l in zip(V, Lambdas0)])
print(c)
X_shape = X_shape - c.flatten() 

plt.plot(*X_shape.T)



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

## Compute PHT
from pbsig.pht import pht_0_dist, pht_preprocess_pc, pht_0
X = [dataset[k] for k in valid_keys]
E = [np.array(list(zip(range(x.shape[0]), np.roll(range(x.shape[0]), -1)))) for x in X]
dgms = [pht_0(x,e,nd=132,transform=False) for x,e in zip(X,E)]
D = pht_0_dist(dgms, E, nd=132, preprocess=False, progress=True, point_cloud=False)

## Is the PHT a discriminator?
assert len(np.unique(D)) == len(D)

## Show PHT distance matrix
from pbsig.color import scale_interval
scaling = 'linear' # linear, equalize, logarithmic
plt.imshow(squareform(scale_interval(D, scaling=scaling)))
plt.suptitle(f"MPEG dataset - PHT")
plt.title(f"{scaling} scaling")

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


## Why are the two chickens and turtles far from each other?
dataset.keys() # d(turtle1, turtle2) > d(turtle1, watch1)?

T1, T2, T3 = dataset[('turtle',1)], dataset[('turtle',2)], dataset[('watch',1)]
# R_refl = np.array([[-1, 0], [0, 1]])
# dataset[('turtle',1)] = T1 = T1 @ R_refl
plt.plot(*T1.T)
plt.scatter(*T1.T, c=T1 @ np.array([0,1]))
plt.plot(*T2.T)
plt.scatter(*T2.T, c=T2 @ np.array([0,1]))
plt.plot(*T3.T)
plt.scatter(*T3.T, c=T3 @ np.array([0,1]))

E = np.array(list(cycle_window(range(T1.shape[0]))))
T1_dgms = pht_0(T1,E,nd=132,transform=False, replace_inf=True)
T2_dgms = pht_0(T2,E,nd=132,transform=False, replace_inf=True)
T3_dgms = pht_0(T3,E,nd=132,transform=False, replace_inf=True)

## First just do the optimal rotation, and let's verify 
# from scipy.spatial import procrustes
from procrustes.rotational import rotational
res = rotational(T1, T2, pad=False, translate=False, scale=False)
plt.plot(*res.new_a.T)
plt.plot(*res.new_b.T)
plt.gca().set_aspect("equal")
res.error

## Manually verify 
rot_mat = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Theta = np.linspace(0, 2*np.pi, 132, endpoint=False)
opt_theta = Theta[np.argmax([np.linalg.norm((T1@rot_mat(t))-T2, 'fro') for t in Theta])]
plt.plot(*(T1@rot_mat(opt_theta)).T)
plt.plot(*T2.T)
plt.gca().set_aspect("equal")

# from scipy.linalg import orthogonal_procrustes
# R, scale = orthogonal_procrustes(T1, T2)
# plt.plot(*(T1 @ R).T)
# plt.plot(*(T2).T)

pht_0_dist([T1_dgms, T2_dgms], preprocess=False, progress=True, diagrams=True)

plot_fake_vines(T1_dgms)
plot_fake_vines(T2_dgms)



def plot_fake_vines(dgms):
  plot_dgm(dgms[0])
  for dgm in dgms:
    plt.scatter(*dgm.T, s=0.05, marker=".")
  min_x = min([min(dgm[:,0]) for dgm in dgms])
  max_x = max([max(dgm[1:,1]) for dgm in dgms])
  plt.gca().set_xlim(min_x, max_x)
  plt.gca().set_ylim(min_x, max_x)
  plt.gca().set_aspect('equal')


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
