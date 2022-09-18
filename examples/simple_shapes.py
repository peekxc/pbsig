import numpy as np
from scipy.spatial.distance import pdist 
from pbsig.PHT import pht_0

## Circle example 
def make_circle(n: int):
  theta = np.linspace(0, 2*np.pi, n)
  S = np.c_[np.cos(theta), np.sin(theta)]
  return(S)

## Triangle example 
def make_triangle(ne: int):
  a,b,c = np.array([-np.sqrt(2)/2, 0]), np.array([np.sqrt(2)/2, 0]), np.array([0.0, 1.0])
  P = []
  A = np.linspace(0, 1, ne+1, endpoint=False)
  P.extend([(1-alpha)*a + alpha*b for alpha in A])
  P.extend([(1-alpha)*b + alpha*c for alpha in A])
  P.extend([(1-alpha)*c + alpha*a for alpha in A])
  return np.vstack(P)

def make_n_gon(sides, radius=1, rotation=0, translation=None):
  from math import pi, sin, cos
  s = pi * 2 / sides # segment
  points = [(sin(s * i + rotation) * radius, cos(s * i + rotation) * radius) for i in range(sides)]
  if translation:
    points = [[sum(pair) for pair in zip(point, translation)] for point in points]
  return(np.array(points))

cycle_outline = lambda X: np.vstack([X, X[[-1,0],:]])
# for n in range(3, 20, 2):
#   #plt.scatter(*make_n_gon(n).T)
#   plt.plot(*cycle_outline(make_n_gon(n)).T)

#S = make_triangle(10)
S = make_n_gon(10)
E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))

from pbsig import plot_dgm
D = pht_0(S, E, 100)

diam = max(pdist(S))
for i in range(len(D)):
  D[i][:,1] = np.clip(D[i][:,1], -np.inf, diam)

import matplotlib.pyplot as plt
from pbsig.color import bin_color
C = bin_color(range(len(D)), x_min=0, x_max=len(D))
for d, pc in zip(D, C):
  plt.scatter(*d.T, color=np.array(pc), alpha=0.80)
plt.gca().set_aspect('equal')
plt.gca().set_xlim(-2.05, 2.05)
plt.gca().set_ylim(-2.05, 2.05)

def largest_births(S):
  E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))
  D = pht_0(S, E, 100)
  diam = max(pdist(S))
  for i in range(len(D)):
    D[i][:,1] = np.clip(D[i][:,1], -np.inf, diam)
  births = np.array([d[np.argmax(np.diff(d, axis=1)),0] for d in D])
  return(births)

for n in range(3, 15, 2):
  births = largest_births(make_n_gon(n))
  plt.plot(births)

## Compare two triangles, rotated out of alignment, with bottleneck distance 
# plt.plot(*cycle_outline(make_n_gon(3, rotation=np.pi/3)).T); plt.plot(*cycle_outline(make_n_gon(3, rotation=0)).T); 
def lerp_refine(X, n: int):
  R = []
  for p,q in pairwise(X):
    T = np.linspace(0, 1, n)
    R.extend([(1-t)*p + t*q for t in T])
  return(np.array(R))

T1 = lerp_refine(make_n_gon(3), 10)

## triangle testing
from pbsig import rotate_S1 
for n in range(3, 1000):
  S = make_n_gon(n)
  F = np.array([fu for fu in rotate_S1(S, n=1000, include_direction=False)])
  plt.plot(F.min(axis=1)) # births
  # F.min(axis=1) # births as function of theta
  # F.min(axis=0)
  # print(f"min: {min(F.min(axis=0))}, max: {max(F.max(axis=0))}")

# E = np.array([[0,1], [1,2], [0,2]])
# u = np.array([0,1])
# u = S[1,:] / np.linalg.norm(S[1,:])
S = make_n_gon(n)
E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))
D = np.array([lower_star_ph_dionysus(fu, E,[]) for fu in rotate_S1(S, n=100, include_direction=False)])

from pbsig.persistence import boundary_matrix
n = 5
S = make_n_gon(n)
E = np.array(list(zip(range(S.shape[0]), np.roll(range(S.shape[0]), -1))))
D0, D1 = boundary_matrix({'vertices': list(range(n)), 'edges': E}, p = (0,1))
fv = S @ np.array([0,1])
fe = fv[E].max(axis=1)
D1 = D1[:, np.argsort(fe)]
#D1 = D1[np.ix_(np.argsort(fv), np.argsort(fe))].tocsc()
D1.sort_indices()
D1.data = np.tile([1,-1], D1.shape[0]) # replace 
R0, R1, V0, V1 = reduction_pHcol(D0, D1)
# persistence_pairs(R0, R1, f=(np.sort(fv), np.sort(fe)), collapse=False)
Fv = dict(zip(range(S.shape[0]), fv))
Fe = dict(zip([str(e) for e in E], fe))
Fv = dict(sorted(Fv.items(), key=lambda kv: (kv[1], kv[0])))
Fe = dict(sorted(Fe.items(), key=lambda kv: (kv[1], kv[0])))
persistence_pairs(R0, R1, f=(Fv, Fe), collapse=False)

lower_star_ph_dionysus(fv, E, [])[0]

## Radial plot 
## birth |-> color in viridis  
## theta |-> angular parameter 
## lerp  |-> radius parameter
import pyflubber
from pbsig.color import bin_color
from pbsig.utility import pht_proprocess_pc
f = pht_proprocess_pc(make_circle(72))
S1, S2 = f(make_circle(72)), f(make_triangle(int(69/3)))

f = pht_proprocess_pc
S1, S2 = f(make_circle(72), transform=True), f(make_triangle(int(69/3)), transform=True)

plt.scatter(*S1.T)
plt.scatter(*S2.T)
plt.gca().set_aspect('equal')
## Todo: center S1 and S2 at same point

E = np.array(list(zip(range(S1.shape[0]), np.roll(range(S1.shape[0]), -1))))
#plt.scatter(*S2.T)
#plt.scatter(*S2.T)

#xmin, xmax = np.min(births), np.max(births)

min_birth, max_birth = np.inf, -np.inf
for i, t in enumerate(np.linspace(0, 1, 50)):
  S = lerp(t)
  diam = max(pdist(S))
  D = pht_0(S, E, 100)
  for j in range(len(D)): 
    D[j][:,1] = np.clip(D[j][:,1], -np.inf, diam)
  births = np.array([d[np.argmax(np.diff(d, axis=1)),0] for d in D])
  min_birth = min([min_birth, np.min(births)])
  max_birth = max([max_birth, np.max(births)])

from pbsig import pairwise
fig = plt.figure(figsize=(5,5), dpi=320)
ax = fig.gca()
lerp = pyflubber.interpolator(S1, S2, closed=True)
#ax.plot(*XY.T, color=C, s=0.45)
for i, t in enumerate(np.linspace(0, 1, 50)):
  S = lerp(t)
  diam = max(pdist(S))
  D = pht_0(S, E, 100)
  for j in range(len(D)): 
    D[j][:,1] = np.clip(D[j][:,1], -np.inf, diam)
  births = np.array([d[np.argmax(np.diff(d, axis=1)),0] for d in D])
  # fv = list(rotate_S1(S, n=100, include_direction=False))[32]
  # births = np.vstack(D)[:,0]
  C = bin_color(births, lb=min_birth, ub=max_birth)
  #scale_interval(births, xmin, xmax)[0]
  theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
  radius = np.repeat(i/50, len(theta))
  XY = np.c_[radius*np.cos(theta), radius*np.sin(theta)]
  #ax.scatter(*XY.T, color=C, s=0.45)
  pts_iter = pairwise(np.vstack([XY, XY[[-1,0],:]]))
  C_iter = np.vstack([C, C[0]])
  for (p,q),c in zip(pts_iter, C_iter):
    ax.plot([p[0], q[0]],[p[1], q[1]],color=c, linewidth=2.00)
  # plt.scatter(*XY.T, color=C)

# plt.scatter(*S.T, c=fv)
# plt.gca().set_aspect('equal')
# fv = list(rotate_S1(S, n=100, include_direction=False))[32]
# lower_star_ph_dionysus(fv, E, [])[0]