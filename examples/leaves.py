# %% Imports
import numpy as np 
from pbsig import * 
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt 

# %% Preprocess data set
base_dir = "/Users/mpiekenbrock/Downloads/"
leaf_types = [1,2] # 3,6,7
leaf_nums = [5,10,15,20] # 21,22,23,24,26
# leaf_fps = ['l1nr071.tif']
LEAF_DATASET = {}
for lt in leaf_types:
  for ln in leaf_nums:
    img_dir = base_dir + f"leaf{lt}/" + f"l{lt}nr{ln:03d}.tif"
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = 180
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = (contours[np.argmax([len(c) for c in contours])])
    S = contours[:,0,:]
    LEAF_DATASET[(lt, ln)] = S

#%% Write the outlines to disk
# x_rng, y_rng, d = img.shape
# img_contours = np.zeros(img.shape)
# cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)
# cv2.imwrite('/Users/mpiekenbrock/Downloads/l1nr071_outline.png',img_contours) 
# plt.plot(*LEAF_DATASET[(1, '15')].T)

# %% Plot leaves 
from itertools import product
NT, NL = len(leaf_types), len(leaf_nums)
fig, axs = plt.subplots(NT, NL, dpi=220)
for (k, S), (i,j) in zip(LEAF_DATASET.items(), product(range(NT), range(NL))):
  axs[i,j].plot(*S.T, linewidth=0.40)
  axs[i,j].axis('off')
  axs[i,j].set_aspect('equal')
  # axs[i,j].set_title(k, fontsize=3)

# %% Show vineyards 


# %% Show discrete vineyards 
from pbsig import rotate_S1
from pbsig.persistence import lower_star_ph_dionysus
from pbsig.betti import lower_star_betti_sig
from pbsig.utility import progressbar, simplify_outline
from pbsig.pht import pht_preprocess_pc

N_SEG = 100
dgms = {}
# for k, S in progressbar(LEAF_DATASET.items(), len(LEAF_DATASET)):
k, S = next(iter(LEAF_DATASET.items()))
S = pht_preprocess_pc(S.astype(float), transform=True)
S = np.array([p.start for p in simplify_outline(S, N_SEG)])
E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))
F = list(rotate_S1(S, 132, include_direction=False))
dgms[k] = []
for f in F:
  dgm0 = lower_star_ph_dionysus(f, E, [])[0]
  dgm0[dgm0[:,1] == np.inf,1] = max(f)
  dgms[k].append(dgm0)

f_min = min([min(d.min(axis=1)) for d in dgms[(1,5)]])
f_max = max([max(d.max(axis=1)) for d in dgms[(1,5)]])
fig, ax = plot_dgm(dgms[(1,5)][0])
ax.set_ylim(f_min*1.10, f_max*1.10)
ax.set_xlim(f_min*1.10, f_max*1.10)
for dgm0 in dgms[(1,5)]:
  plt.scatter(*dgm0.T, s=0.05)

## Does the PHT work????
from pbsig.pht import pht_0_dist
# k, S = next(iter(LEAF_DATASET.items()))

# S = pht_preprocess_pc(S.astype(float), transform=True)
# S = np.array([p.start for p in simplify_outline(S, N_SEG)])
from scipy.spatial.distance import pdist, squareform
from pbsig.color import scale_interval
outline = lambda n: np.array(list(zip(range(n), np.roll(range(n), -1))))
edges = [outline(S.shape[0]) for S in LEAF_DATASET.values()]
shapes = [S.astype(float) for S in LEAF_DATASET.values()]
D = pht_0_dist(shapes, edges, nd=32, preprocess=True, progress=True)


D_PHT = squareform(D)
scaling = 'linear' # linear, equalize, logarithmic
plt.imshow(scale_interval(D_PHT, scaling=scaling))
plt.suptitle(f"Leaf dataset - PHT")
plt.title(f"{scaling} scaling")
plt.colorbar()

# shape_center
# for k in range(1, 100, 5):
#   theta = np.linspace(0, 2*np.pi, 2*k, endpoint=False)
#   V = np.c_[np.cos(theta), np.sin(theta)]
#   print(np.array([min(X @ vi)*np.array(vi) for vi in V]).mean(axis=0))

# plt.plot(*S.T)
# plt.gca().set_aspect('equal')
# plt.scatter(111.50179484,375.30621169, c='red')


# %% 
from pbsig import rotate_S1
from pbsig.betti import lower_star_betti_sig
from pbsig.utility import progressbar, simplify_outline
from pbsig.pht import pht_preprocess_pc

N_SEG = 100
omega = 0.85 # window size 
sigs = []
for S in progressbar(LEAF_DATASET.values(), len(LEAF_DATASET)):
  S = pht_preprocess_pc(S.astype(float), transform=True)
  S = np.array([p.start for p in simplify_outline(S, N_SEG)])
  E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))
  F = list(rotate_S1(S, 32, include_direction=False))
  birth_lb, birth_ub = min([min(f) for f in F]), max([min(f) for f in F])
  a, b = 2*birth_lb, 5*(2*birth_lb + (birth_ub-birth_lb)/2)
  sigs.append(lower_star_betti_sig(F, E, nv=S.shape[0], a=a, b=b, method = "nuclear", w=omega))

plt.plot(*S.T)
plt.gca().set_aspect('equal')

len(np.unique(pdist(np.vstack(sigs)))) == int(len(sigs)*(len(sigs)-1)/2)
from scipy.spatial.distance import pdist, squareform
from pbsig.color import scale_interval
D = squareform(pdist(np.vstack(sigs)))
scaling = 'linear' # equalize, logarithmic
plt.imshow(scale_interval(D, scaling=scaling))
plt.suptitle(f"Leaf dataset - PBT - Nuclear, a={a:0.2f},b={b:0.2f},w={omega:0.2f}")
plt.title(f"{scaling} scaling")
plt.colorbar()


from pbsig.persistence import lower_star_dionysuss
