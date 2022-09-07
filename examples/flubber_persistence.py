import re
import numpy as np
import matplotlib.pyplot as plt
import pyflubber
from PIL import Image, ImageFilter
from itertools import chain, product
from svgpathtools import svg2paths

from typing import * 
from numpy.typing import ArrayLike
from pbsig.utility import PL_path, shape_center, pairwise
from pbsig import uniform_S1

complex2points = lambda x: np.c_[np.real(x), np.imag(x)]

# fish1 = Image.open("/Users/mpiekenbrock/Downloads/fish1.png")
# Image.open("/Users/mpiekenbrock/Downloads/shark.svg")
# fish2 = Image.open("/Users/mpiekenbrock/Downloads/fish2.png")
# fish1_outline = fish1.filter(ImageFilter.FIND_EDGES)

folder = "/Users/mpiekenbrock/Downloads/"
SVGs = ["shark.svg", "fish1.svg"]
V_dir = np.array(list(uniform_S1(10)))
SVG_pts = []
for svg in SVGs: 
  paths, attributes = svg2paths(folder + svg)
  ## simplify to PL-set of Lines
  #C = offset_curve(paths[0], 0.01, steps = 3)
  C = PL_path(paths[0], k=450) 
  P = complex2points([c.start for c in C])
  u = shape_center(P, method="directions", V=V_dir)
  P = P - u 
  L = sum([-np.min(P @ vi[:,np.newaxis]) for vi in V_dir])
  P = (1/L)*P
  SVG_pts.append(P)

## Plot super-imposed shapes
for S in SVG_pts:
  plt.plot(*S.T)
plt.gca().set_xlim(-0.5,0.5)
plt.gca().set_ylim(-0.5,0.5)
plt.gca().set_aspect('equal')

from easing_functions import ExponentialEaseOut, LinearInOut
#Ease = ExponentialEaseOut(start=0, end=1.0, duration=1.0)
Ease = LinearInOut(start=0, end=1.0, duration=1.0)
#F_interp = pyflubber.interpolator(SVG_pts[0], SVG_pts[1], closed=False)
F_interp = pyflubber.interpolator(shapes[0], shapes[1], closed=True)
time = np.linspace(0, 1, 4*8, endpoint=True)

fig, axs = plt.subplots(4, 8, figsize=(12,4))
for t, (i, j) in zip(time, product(range(4), range(8))):
  Pt = F_interp(Ease(t))
  axs[i,j].plot(*Pt.T)
  axs[i,j].axis('off')



# new_path = PL_path(paths[0], 16)
# disvg(new_path)



from pbsig.persistence import lower_star_ph_dionysus
from itertools import chain
from pbsig import plot_dgm, rotate_S1

Pt = F_interp(Ease(0))
E = np.array(list(chain(pairwise(range(Pt.shape[0])), iter([(Pt.shape[0]-1, 0)]))))
dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(Pt, n=100, include_direction=False)]

## Filtration values
F = [fv for fv in rotate_S1(Pt, n=100, include_direction=False)]

import ot
from gudhi.wasserstein import wasserstein_distance

from pbsig import plot_dgm
u = np.array([1,1])
u = u/np.linalg.norm(u)
proj_u = lambda x: np.dot(x, u)*u

d0, d1 = dgms0[0], dgms0[1]
wd, wm = wasserstein_distance(d0, d1, matching=True, order=2.0, keep_essential_parts=True)

## Plot diagram + matching
if wm is not None:
  fig, ax = plot_dgm(d0)
  ax.scatter(*d1.T, s=0.15, zorder=0, c='orange', alpha=1.0)
  for i,j in wm:
    if i != -1 and j == -1:
      L = np.vstack((d0[i,:], proj_u(d0[i,:]))) 
    elif i == -1 and j != -1:
      L = np.vstack((d1[j,:], proj_u(d1[j,:]))) 
    else:
      L = np.vstack((d0[i,:], d1[j,:]))
    #d += np.linalg.norm(L[1,:]-L[0,:])**2
    ax.plot(*L.T, linewidth=0.60, c='blue', zorder=-100, alpha=0.50) 

  max_f = 1.05*max(chain(d0[:,1], d1[:,1]), key=lambda x: 0 if x == np.inf else x)
  ax.set_aspect('equal')
# ax.set_xlim(0, max_f)
# ax.set_ylim(0, max_f)

## Empirically use discretization to determine PHT-distance
D = [] # space of persistence diagrams
for t in np.linspace(0, 1, 12):
  Pt = F_interp(Ease(t))
  E = np.array(list(chain(pairwise(range(Pt.shape[0])), iter([(Pt.shape[0]-1, 0)]))))
  dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(Pt, n=100, include_direction=False)]
  D.append(dgms0)

def W2_mod_rot(D0, D1) -> float:
  assert len(D0) == len(D1), "Lists containing diagrams should be equal"
  w2 = lambda a,b: wasserstein_distance(a, b, matching=False, order=2.0, keep_essential_parts=True)
  wdists = []
  m = len(D0)
  for i in range(m): 
    D_rot = D0[(m-i):] + D0[i:]  
    wdists.append(sum([w2(d0,d1) for d0,d1 in zip(D_rot, D1)]))
  return(wdists[np.argmin(wdists)])

from math import comb
from itertools import combinations
from pbsig.utility import progressbar
pht_dist = np.array([W2_mod_rot(D0, D1) for D0, D1 in progressbar(combinations(D, 2), comb(len(D), 2))])

## It works for the interpolation!
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
plt.imshow(squareform(pht_dist))


## Now try with many SVG images
from pbsig.datasets import animal_svgs
dataset = animal_svgs()

# from svgpathtools import wsvg
# wsvg(dataset['cooki'], filename="../src/pbsig/data/animal_svgs/cooki.svg")

def pht_preprocess(nd: int = 32, resolution: int = 100):
  complex2points = lambda x: np.c_[np.real(x), np.imag(x)]
  V_dir = np.array(list(uniform_S1(nd)))
  def _preprocess(path):
    C = PL_path(path, k=resolution) 
    P = complex2points([c.start for c in C])
    u = shape_center(P, method="directions", V=V_dir)
    P = P - u 
    L = sum([-np.min(P @ vi[:,np.newaxis]) for vi in V_dir])
    P = (1/L)*P
    return(P)
  return _preprocess

PHT_pre = pht_preprocess(nd=50, resolution=300)

## Preprocess via translation, scaling, PL-outline simplification, etc
shapes = [PHT_pre(d) for d in dataset.values()]

## Visualize shapes
fig, axs = plt.subplots(2,4, figsize=(8,4), dpi=320)
for (i,j),S in zip(product(range(2),range(4)), shapes):
  axs[i,j].plot(*S.T, linewidth=0.92)
  axs[i,j].axis('off')

## Get 2-wasserstein to compare
D = [] # space of persistence diagrams
for P in shapes:
  E = np.array(list(chain(pairwise(range(P.shape[0])), iter([(P.shape[0]-1, 0)])))) # outline
  dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(P, n=32, include_direction=False)]
  D.append(dgms0)
pht_dist = np.array([W2_mod_rot(D0, D1) for D0, D1 in progressbar(combinations(D, 2), comb(len(D), 2))])

plt.imshow(squareform(pht_dist))



## time / lifetime plot
X = [abs(dgm[:,1]-dgm[:,0]) for dgm in dgms0]
for i, t in enumerate(time):
  xi = list(filter(lambda x: x != np.inf, X[i]))
  plt.scatter(np.repeat(t, len(xi)), xi, s=0.45)
plt.gca().set_ylim(-1, 25)


## TODO: 
## 1. make a signature set of vines for each shape
## 2. Compute minimum integrated wasserstein distance controlling for rotation 
## 3. Form a distance matrix of the IWD for each pair of outlines 
from pbsig.persistence import boundary_matrix

## (0) Acquire shape information 
shape1 = SVG_pts[0]
V = np.fromiter(range(shape1.shape[0]),dtype=int)
E = np.array(list(chain(pairwise(range(shape1.shape[0])), iter([(shape1.shape[0]-1, 0)])))) # edges
K = { 'vertices': V, 'edges': E }

## (1) Form a signature set of vines for each shape
# np.arccos(np.dot(Dir_vectors[:,2], Dir_vectors[:,1]))
Filt_values = np.vstack([fv for fv, vi in rotate_S1(shape1, n=16)])
Dir_vectors = np.hstack([vi for fv, vi in rotate_S1(shape1, n=16)])
for i, (fv0, fv1) in enumerate(pairwise(Filt_values)):
  if i == 0: 
    D0, D1 = boundary_matrix(K, p=(0,1))
    R0, R1, V0, V1 = reduction_pHcol(D0, D1)
    Vf = dict(sorted(zip(to_str(V), fv0), key=index(1)))
    Ef = dict(sorted(zip(to_str(E), fe0), key=index(1)))

    D1, D2 = boundary_matrix(K, p=(1,2), f=((fv0,fe0), (fe0,ft0)))
    D1, D2 = D1.tolil(), D2.tolil()
    

    ## Handle deficient cases
    R0, V0 = sps.lil_matrix((0,len(V))), sps.lil_matrix(np.eye(len(V)))
    R3, V3 = sps.lil_matrix((len(T), 0)), sps.lil_matrix((0,0))
    D0, D3 = R0, R3
  else: 
    fe0 = fv0[E].max(axis=1)
    fe1 = fv1[E].max(axis=1)
    tr = linear_homotopy(fe0, fe1, plot=False, interval=[0, 0.392])

shape2 = SVG_pts[1]





## vineyards 
from pbsig.persistence import barcodes
d0, d1 = dgms0[0], dgms0[1]

# U0, U1 = np.unique(F[0]), np.unique(F[1])
# d0_I = np.array([(int(np.argmin(abs(a - U0))), int(np.argmin(abs(b - U0)))) for (a,b) in d0], dtype=int)
# d1_I = np.array([(int(np.argmin(abs(a - U1))), int(np.argmin(abs(b - U1)))) for (a,b) in d1], dtype=int)

# ## Matching is *not* invariant under rank-preserving transformations! 
# wm_a = wasserstein_distance(d0, d1, matching=True, order=1.0, keep_essential_parts=True)[1]
# wm_b = wasserstein_distance(d0_I, d1_I, matching=True, order=1.0, keep_essential_parts=True)[1]








## Compare two individual shapes 
from pbsig import rotate_S1, progressbar
from pbsig.betti import lower_star_betti_sig
a,b = 0,0
for S in shapes:
  E = np.array(list(chain(pairwise(range(S.shape[0])), iter([(S.shape[0]-1, 0)])))) # edges
  a += np.mean([min(f) for f in rotate_S1(S, n=100, include_direction=False)])*2
  b += np.mean([max(f) for f in rotate_S1(S, n=100, include_direction=False)])/2
a,b = a/2,b/2

plt.scatter(*S.T)

max([max(pdist(S)) for S in shapes])


S = shapes[0]
S += np.random.normal(size=S.shape, loc=0.0, scale=0.0005)
plt.scatter(*S.T)

sigs = []
for S in shapes:
  S += np.random.normal(size=S.shape, loc=0.0, scale=0.0005)
  E = np.array(list(chain(pairwise(range(S.shape[0])), iter([(S.shape[0]-1, 0)])))) # edges
  FH = progressbar(rotate_S1(S, n=50, include_direction=False), 50)
  # w=0.5, epsilon=0.1 <=> differences ~ 0.20, ranges around 192
  sigs.append(lower_star_betti_sig(FH, p_simplices=E, nv=S.shape[0], a=a,b=b,w=0.5,epsilon=0.1))
# plt.plot(PB_sig0)

## Curves 
shape_names = np.array(list(dataset.keys()))
for sig, sn in zip(sigs[:6], shape_names[:6]): 
  plt.plot(sig, label=sn)
plt.legend()

from itertools import combinations
from scipy.spatial.distance import squareform

min_rot_euc = lambda s0, s1: min([sum(abs(np.roll(s0, i) - s1)) for i in range(len(s0))])

pbn_dist = np.array([min_rot_euc(s0, s1) for (s0, s1) in combinations(sigs, 2)])
ind = [0,1,2,3,4,5,6]
# [np.ix_(ind, ind)]
plt.imshow(squareform(pbn_dist)[np.ix_(ind, ind)])
# plt.legend(np.array(shape_names)[ind])

plt.plot(sigs[0])
plt.gca().set_ylim(0,5)
plt.plot(sigs[1])
plt.gca().set_ylim(0,5)

plt.plot(sum(abs(sigs[0]-sigs[1])))

from pbsig.utility import isotonic_regress

isotonic_regress(t)

from easing_functions import LinearInOut
Ease = LinearInOut(start=0, end=1.0, duration=1.0)
F_interp = pyflubber.interpolator(shapes[0], shapes[1], closed=True)

plt.scatter(*F_interp(0.20).T)

from scipy.optimize import minimize
S = shapes[0]
E = np.array(list(chain(pairwise(range(S.shape[0])), iter([(S.shape[0]-1, 0)])))) # outline
dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(P, n=32, include_direction=False)]

fig, ax = plot_dgm(dgms0[0])
for d in dgms0[1:]:
  ax.scatter(*d.T, s=0.15)
ax.autoscale(tight=True)

## Get initial "iterate" for a pair of shapes 
from math import comb
F_interp = pyflubber.interpolator(shapes[0], shapes[1], closed=True)
T = np.linspace(0, 1, 10)
sigs = []
E = np.array(list(chain(pairwise(range(shapes[0].shape[0])), iter([(shapes[0].shape[0]-1, 0)])))) # edges

for i, t in progressbar(enumerate(T), len(T)): 
  S = F_interp(t)
  F = rotate_S1(S, n=25, include_direction=False)
  sigs.append(lower_star_betti_sig(F, p_simplices=E, nv=S.shape[0], a=a,b=b,w=1.15,epsilon=1.50))

from scipy.spatial.distance import squareform
pbn_sig_dm = squareform([min_rot_euc(s0, s1) for (s0, s1) in combinations(sigs, 2)])
plt.imshow(pbn_sig_dm)

## The cost function to minimize (scaling?)
TA = toeplitz(isotonic_regress(project_toeplitz(pbn_sig_dm)))
np.linalg.norm(pbn_sig_dm-TA)

plt.imshow(TA)
# for i, (ti, tj) in progressbar(enumerate(combinations(T, 2)), comb(len(T), 2)):
#   Si, Sj = F_interp(ti), F_interp(tj)
#   Fi = rotate_S1(Si, n=25, include_direction=False)
#   Fj = rotate_S1(Sj, n=25, include_direction=False)
#   Bi = lower_star_betti_sig(Fi, p_simplices=E, nv=Si.shape[0], a=a,b=b,w=0.05,epsilon=0.01)
#   Bj = lower_star_betti_sig(Fj, p_simplices=E, nv=Sj.shape[0], a=a,b=b,w=0.05,epsilon=0.01)
#   betti_diff[i] = sum(abs(Bi - Bj)**2)


## Toeplitz learning idea



