
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pyflubber
from PIL import Image, ImageFilter
from itertools import chain, product
from svgpathtools import svg2paths

complex2points = lambda x: np.c_[np.real(x), np.imag(x)]

P = np.array(list((i,j) for i,j in product(range(3), range(3))))


pyflubber.interpolate()




fish1 = Image.open("/Users/mpiekenbrock/Downloads/fish1.png")
fish2 = Image.open("/Users/mpiekenbrock/Downloads/fish2.png")

fish1_outline = fish1.filter(ImageFilter.FIND_EDGES)





folder = "/Users/mpiekenbrock/Downloads/"
SVGs = ["shark.svg", "fish1.svg"]

SVG_pts = []
for svg in SVGs: 
  paths, attributes = svg2paths(folder + svg)
  C = offset_curve(paths[0], 0.01, steps = 3)
  P = complex2points([c.start for c in C])
  SVG_pts.append(P)

plt.plot(*SVG_pts[0].T)

from easing_functions import *
E = ExponentialEaseOut(start=0, end=1.0, duration=1.0)
F = pyflubber.interpolator(SVG_pts[0], SVG_pts[1])
T = np.linspace(0, 1, 4*8, endpoint=True)

fig, axs = plt.subplots(4, 8, figsize=(12,4))
for t, (i, j) in zip(T, product(range(4), range(8))):
  Pt = F(E(t))
  axs[i,j].plot(*Pt.T)
  axs[i,j].axis('off')


from svgpathtools import parse_path, Line, Path, wsvg
def offset_curve(path, offset_distance, steps=1000):
  """Takes in a Path object, `path`, and a distance,
  `offset_distance`, and outputs an piecewise-linear approximation 
  of the 'parallel' offset curve."""
  nls = []
  for seg in path:
    ct = 1
    for k in range(steps):
      t = k / steps
      offset_vector = offset_distance * seg.normal(t)
      nl = Line(seg.point(t), seg.point(t) + offset_vector)
      nls.append(nl)
  connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
  if path.isclosed():
    connect_the_dots.append(Line(nls[-1].end, nls[0].end))
  offset_path = Path(*connect_the_dots)
  return offset_path



from pbsig.persistence import lower_star_ph_dionysus
from itertools import chain
from pbsig import plot_dgm, rotate_S1
def pairwise(C): return(zip(C, C[1:]))

E = np.array(list(chain(pairwise(range(Pt.shape[0])), iter([(Pt.shape[0]-1, 0)]))))
dgms0 = [lower_star_ph_dionysus(fv, E, [])[0] for fv in rotate_S1(Pt, n=100, include_direction=False)]

## Filtration values
F = [fv for fv in rotate_S1(Pt, n=100, include_direction=False)]

import ot as ot
from gudhi.wasserstein import wasserstein_distance

from pbsig import plot_dgm
u = np.array([1,1])
u = u/np.linalg.norm(u)
proj_u = lambda x: np.dot(x, u)*u

d0, d1 = dgms0[0], dgms0[1]
wd, wm = wasserstein_distance(dgms0[0], dgms0[1], matching=True, order=2.0, keep_essential_parts=True)

## Plot diagram + matching
fig, ax = plot_dgm(d0)
ax.scatter(*d1.T, s=0.15, zorder=0, c='orange')
for i,j in wm:
  if i != -1 and j == -1:
    L = np.vstack((d0[i,:], proj_u(d0[i,:]))) 
  elif i == -1 and j != -1:
    L = np.vstack((d1[j,:], proj_u(d1[j,:]))) 
  else:
    L = np.vstack((d0[i,:], d1[j,:]))
  #d += np.linalg.norm(L[1,:]-L[0,:])**2
  ax.plot(*L.T, linewidth=0.60, c='blue', zorder=-100) 

## time / lifetime plot
X = [abs(dgm[:,1]-dgm[:,0]) for dgm in dgms0]
for i, t in enumerate(T):
  xi = list(filter(lambda x: x != np.inf, X[i]))
  plt.scatter(np.repeat(t, len(xi)), xi, s=0.45)
plt.gca().set_ylim(-1, 25)



## vineyards 
from pbsig.persistence import barcodes
d0, d1 = dgms0[0], dgms0[1]

# U0, U1 = np.unique(F[0]), np.unique(F[1])
# d0_I = np.array([(int(np.argmin(abs(a - U0))), int(np.argmin(abs(b - U0)))) for (a,b) in d0], dtype=int)
# d1_I = np.array([(int(np.argmin(abs(a - U1))), int(np.argmin(abs(b - U1)))) for (a,b) in d1], dtype=int)

# ## Matching is *not* invariant under rank-preserving transformations! 
# wm_a = wasserstein_distance(d0, d1, matching=True, order=1.0, keep_essential_parts=True)[1]
# wm_b = wasserstein_distance(d0_I, d1_I, matching=True, order=1.0, keep_essential_parts=True)[1]