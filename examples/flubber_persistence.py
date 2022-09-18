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










# %% Toeplitz idea
from pbsig.datasets import animal_svgs
from pbsig.utility import pht_preprocess_path_2d
animal_paths = animal_svgs()
PHT_pre = pht_preprocess_path_2d(n_directions=50, n_segments=50)

## Preprocess via translation, scaling, PL-outline simplification, etc
shapes = [PHT_pre(p) for p in animal_paths.values()]
nv = shapes[0].shape[0]
E = np.array(list(chain(pairwise(range(nv)), iter([(nv-1, 0)])))) # edges for all shapes are identical

# %% Visualize shapes
fig, axs = plt.subplots(2,4, figsize=(8,4), dpi=320)
for (i,j),S in zip(product(range(2),range(4)), shapes):
  axs[i,j].plot(*S.T, linewidth=0.92)
  axs[i,j].axis('off')

# %% Get effective bounds on all the parameters 

## Bounds on birth/death: since shape is centered at origin, smallest/largest projection 
## onto any unit vector 'v' occurs when 'v' is parallel with the point w/ largest norm
max_f = np.max(np.linalg.norm(shapes[0], axis=1))
a_min, b_max = -max_f, max_f

## Effective bounds on epsilon/omega
## omega could be infinite in principle; effective bound (where no values are clamped) is [0, 2|b-a|]
## same principle applies to epsilon; effective bound is where phi_eps(M) \in [rank(M), 1] [0, max_spectral_norm(A)]
from pbsig.utility import spectral_bound
omega_min, omega_max = np.finfo(float).eps, abs(b_max-a_min)
epsilon_min, epsilon_max = np.finfo(float).eps, spectral_bound(range(nv), E, "laplacian")
# D1 = boundary_matrix({'vertices': range(nv), 'edges':E}, p=1)
# np.max(np.linalg.eigh(D1.A @ D1.T)[0])


# %% Do a parameter space search 

## Persistent betti transform for n directions
def pbt_outline(S: ArrayLike, n_dir: int, **kwargs):
  m = S.shape[0]
  E = np.array(list(chain(pairwise(range(m)), iter([(m-1, 0)])))) # edges
  F = rotate_S1(S, n=n_dir, include_direction=False)
  return(lower_star_betti_sig(F, p_simplices=E, nv=m, **kwargs))

## Choose the two shapes to look at 
S1, S2 = shapes[0], shapes[1]
F_interp = pyflubber.interpolator(S1, S2, closed=True)

## Visualize 
time = np.linspace(0, 1, 4*8, endpoint=True)
fig, axs = plt.subplots(4, 8, figsize=(12,4))
for t, (i, j) in zip(time, product(range(4), range(8))):
  Pt = F_interp(t)
  axs[i,j].plot(*Pt.T)
  axs[i,j].axis('off')

## 
from itertools import product
n_params = 10
rng = lambda a,b: np.linspace(a,b,n_params)

R = {}
for t in np.linspace(0, 1, 10):
  S = F_interp(t)
  for cc, (a,b,w,eps) in progressbar(enumerate(product(rng(a_min, b_max), rng(a_min, b_max), rng(omega_min, omega_max), rng(epsilon_min, epsilon_max))), count=n_params**4):
    if a < b:
      R[(t, cc)] = {
        'params' : (a,b,w,eps), 
        'signature' : pbt_outline(S, n_dir=15, a=a, b=b, w=w, epsilon=eps)
      }

# def save_object(obj, filename):
#   import pickle
#   with open(filename, 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
# save_object(R, "shape_signatures.pickle")
## Get parameter ranges
a_rng = rng(a_min, b_max)
b_rng = rng(a_min, b_max)
w_rng = rng(omega_min, omega_max)
e_rng = rng(epsilon_min, epsilon_max)

def extract_pbn(x0:int, x1:int, x2:int, x3:int):
  assert x0 < x1, "invalid"
  a = a_rng[x0]
  b = b_rng[x1]
  w = w_rng[x2]
  e = e_rng[x3]
  vals = list(filter(lambda kv: kv[1]['params'] == (a,b,w,e), R.items()))
  return({ k[0]: v['signature'] for k,v in vals})
    
#   s0_keys = list(filter(lambda k: k[1] > 0 and k[0] == 0.0, R.keys()))

# s0_keys = list(filter(lambda k: k[1] > 0 and k[0] == 0.0, R.keys()))
# S0_betti0 = list(filter(lambda v: v['params'][2] == w_rng[0] and v['params'][3] == e_rng[0], [R[k] for k in s0_keys]))
# S0_sigs = [b['signature'] for b in S0_betti0]

# s1_keys = list(filter(lambda k: k[1] > 0 and k[0] == 1.0, R.keys()))
# S1_betti0 = list(filter(lambda v: v['params'][2] == w_rng[0] and v['params'][3] == e_rng[0], [R[k] for k in s1_keys]))
# S1_sigs = [b['signature'] for b in S1_betti0]


extract_pbn(0,1,0,0).values()

def pbn_dist(sigs):
  from scipy.spatial.distance import squareform
  from itertools import combinations
  min_rot_euc = lambda s0, s1: min([sum(abs(np.roll(s0, i) - s1)) for i in range(len(s0))])
  pbn_sig_dm = squareform([min_rot_euc(s0, s1) for (s0, s1) in combinations(sigs, 2)])
  return(pbn_sig_dm)

fig, axs = plt.subplots(4,4, figsize=(8,8),dpi=320)
for (i,j),(w,e) in zip(product(range(4), range(4)), product(w_rng[[0,2,4,6]], e_rng[[0,2,4,6]])):
  axs[i,j].imshow(pbn_dist(extract_pbn(6,8,i,j).values()))
  axs[i,j].axis('off')
  axs[i,j].title.set_text(f"(w:{w:.2f}, e:{e:.2f}) --- (a:{6}, b:{8})")
  axs[i,j].title.set_size(5)
  #plt.title("eps")


from pbsig.color import bin_color


# min_util, max_util = np.inf, -np.inf
max_util = np.max(list(all_toeplitz_dists.values()))
min_util = np.min(list(all_toeplitz_dists.values()))

fig, axs = plt.subplots(4,4, figsize=(8,8),dpi=320)
for (i,j),(w,e) in zip(product(range(4), range(4)), product(w_rng[[0,2,4,6]], e_rng[[0,2,4,6]])):
  k,l = list(w_rng).index(w), list(e_rng).index(e)
  AB_toeplitz = {}
  for a,b in combinations(a_rng, 2):
    if a < b:
      ai,bj = list(a_rng).index(a), list(b_rng).index(b)
      dm = pbn_dist(extract_pbn(ai,bj,k,l).values())
      TA = toeplitz(isotonic_regress(project_toeplitz(dm)))
      AB_toeplitz[(a,b)] = np.linalg.norm(dm-TA)
  utility_metric = np.array(list(AB_toeplitz.values()))
  # min_util, max_util = min(utility_metric), max(utility_metric)
  # min_util = min(min_util, min(utility_metric))
  # max_util = max(max_util, max(utility_metric))
  for (a,b), u in AB_toeplitz.items():
    axs[i,j].scatter(a,b,color=bin_color([u], min_x=min_util, max_x=max_util, scaling="logarithmic"))
  axs[i,j].axis('off')
  axs[i,j].title.set_text(f"Toeplitz global utility --- w:{w:.2f}({k}), e:{l:.2f}({l})")
  axs[i,j].title.set_size(5)

# fig = plt.figure(figsize=(8,8), dpi=320)
# ax = plt.gca()
  
## Look at persistenc ediagrams of interolation 
fig = plt.figure(figsize=(8,8), dpi=320)
# fig, axs = plt.subplots(2,5, figsize=(8,4),dpi=320)
ax = fig.gca()
ax.set_xlim(min(a_rng)*0.95, max(a_rng)*1.05)
ax.set_ylim(min(a_rng)*0.95, max(a_rng)*1.05)
T = np.linspace(0, 1, 10)
for t, (i,j) in zip(T, product(range(2), range(5))):
  S = F_interp(t)
  E = np.array(list(chain(pairwise(range(S.shape[0])), iter([(S.shape[0]-1, 0)])))) # outline
  dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(S, n=15, include_direction=False)]
  for dgm in dgms0:
    ax.scatter(*dgm.T, color=bin_color([t],min_x=0.0, max_x=1.0), s=0.65)
  #   axs[i,j].scatter(*dgm.T, color=bin_color([t],min_x=0.0, max_x=1.0), s=0.45)
  # axs[i,j].set_xlim(min(a_rng)*0.95, max(a_rng)*1.05)
  # axs[i,j].set_ylim(min(a_rng)*0.95, max(a_rng)*1.05)
  # axs[i,j].axis('off')
plt.title("All PHT-dgm0s, color == time / interpolation")

all_toeplitz_dists = {}
for i,j,k,l in progressbar(product(range(10), range(10), range(10), range(10)), 10**4):
  if i < j:
    dm = pbn_dist(extract_pbn(i,j,k,l).values())
    TA = toeplitz(isotonic_regress(project_toeplitz(dm)))
    all_toeplitz_dists[(i,j,k,l)] = np.linalg.norm(dm-TA)


best_ind = list(all_toeplitz_dists.keys())[np.argmax(list(all_toeplitz_dists.values()))]


plt.hist(all_toeplitz_dists.values())
plt.title("all toeplitz distances")

time = np.linspace(0, 1, 4*8, endpoint=True)

  
# for i, t in progressbar(enumerate(T), len(T)): 
#   S = F_interp(t)
#   F = rotate_S1(S, n=25, include_direction=False)
#   sigs.append()

# from scipy.spatial.distance import squareform
# pbn_sig_dm = squareform([min_rot_euc(s0, s1) for (s0, s1) in combinations(sigs, 2)])




# from sympy import FunctionMatrix, symbols, Lambda, MatPow
# from sympy import Function
# i, j, n, m = symbols('i,j,n,m')
# X = FunctionMatrix(3, 3, 'lambda i,j: i+j')

# import sympy as sp
# a, b, c, d = sp.symbols("a b c d")
# B = sp.Matrix([
#   [(a*c),(b**2)],[(b*d),(d*a)]
# ])


# %% 
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



from scipy.spatial.distance import pdist
from pbsig.utility import unrank_C2
D = pdist(shapes[0])
i,j = unrank_C2(np.argmax(D), shapes[0].shape[0])
a_min, b_max = D[i], D[j]










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




# %%
