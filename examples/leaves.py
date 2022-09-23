import numpy as np 
from pbsig import * 
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt 

base_dir = "/Users/mpiekenbrock/Downloads/"
leaf_types = [1,2,3,6,9]
leaf_nums = [5, 10, 15, 20]
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

## Write the outlines to disk
# x_rng, y_rng, d = img.shape
# img_contours = np.zeros(img.shape)
# cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)
# cv2.imwrite('/Users/mpiekenbrock/Downloads/l1nr071_outline.png',img_contours) 
# plt.plot(*LEAF_DATASET[(1, '15')].T)
from pbsig import rotate_S1
from pbsig.betti import lower_star_betti_sig
from pbsig.utility import pht_proprocess_pc, progressbar
from pbsig.utility import simplify_outline

omega = 0.85
sigs = []
for S in progressbar(LEAF_DATASET.values(), len(LEAF_DATASET)):
  S = pht_proprocess_pc(S.astype(float), transform=True)
  S = np.array([p.start for p in simplify_outline(S, 100)])
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

from itertools import product
fig, axs = plt.subplots(5,4, dpi=220)
for (k, S), (i,j) in zip(LEAF_DATASET.items(), product(range(5), range(4))):
  axs[i,j].plot(*cycle_outline(S).T, linewidth=0.40)
  axs[i,j].axis('off')
  axs[i,j].set_aspect('equal')
  # axs[i,j].set_title(k, fontsize=3)




from pbsig.persistence import lower_star_dionysuss
