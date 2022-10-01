import numpy as np 
from pbsig import * 
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt 
from os.path import exists
from PIL import Image


base_dir = "/Users/mpiekenbrock/Downloads/original"
shape_types = ["turtle", "watch", "chicken", "bird", "lizzard", "bone", "bell"]
shape_nums = [1,2,3,4,5]

normalize = lambda X: (X - np.min(X))/(np.max(X)-np.min(X))*255

def largest_contour(img: ArrayLike, threshold: int = 180):
  _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = (contours[np.argmax([len(c) for c in contours])])
  S = contours[:,0,:]
  return(S)

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
    S_path = simplify_outline(S, 500)
    dataset[(st, sn)] = np.array([l.start for l in S_path])

from pbsig.pht import pht_0_dist
from itertools import chain
from pbsig.utility import pairwise
cycle = lambda n: chain(range(n), [0])
cycle_outline = lambda n: chain(pairwise(range(n)), [(0,n-1)])

### Restrict keys 
classes = ["turtle", "chicken", "bone", "bird"]
valid_keys = list(filter(lambda k: k[0] in classes, dataset.keys()))
X = [dataset[k] for k in valid_keys]
E = [np.array(list(cycle_outline(x.shape[0]))) for x in X]
D = pht_0_dist(X, E, nd=32, preprocess=True, progress=True)

## Is the PHT a discriminator?
assert len(np.unique(D)) == len(D)

## Show PHT distance matrix
plt.imshow(squareform(D))

## Show the shapes
fig, axs = plt.subplots(len(classes),5,figsize=(16,6), dpi=220)
for i, class_type in enumerate(classes):
  for j, k in enumerate(filter(lambda k: k[0] == class_type, valid_keys)):
    S = dataset[k]
    axs[i,j].plot(*S[np.fromiter(cycle(S.shape[0]), dtype=int),:].T, linewidth=0.50)
    axs[i,j].axis('off')

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
