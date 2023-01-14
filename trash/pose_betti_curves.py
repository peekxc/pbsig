from numpy.typing import ArrayLike
import numpy as np
from scipy.spatial import Delaunay
from betti import *

## Torus 
nr, nc = 8, 12 # N = np * nr
R, r = 2.4, 0.75 # dist to tube center, tube radius
u,v = np.linspace(0, 2*np.pi, nc), np.linspace(0,2*np.pi,nr)
u,v = np.meshgrid(u,v)
u,v = u.flatten(), v.flatten()
tp = R + r*np.cos(v)
x, y, z = tp*np.cos(u), tp*np.sin(u), r*np.sin(v)
del_tri = Delaunay(np.c_[u,v])
V, T = scale_diameter(np.c_[x,y,z]), del_tri.simplices

## Mobius band 
nr, nc = 8, 12 # N = np * nr
u,v = np.linspace(0, 2*np.pi, nc), np.linspace(-1,1,nr)
u,v = np.meshgrid(u,v)
u,v = u.flatten(), v.flatten()
tp = 1+0.5*v*np.cos(u/2.)
x, y, z = tp*np.cos(u), tp*np.sin(u), 0.5*v*np.sin(u/2.)
del_tri = Delaunay(np.c_[u,v])
V, T = scale_diameter(np.c_[x,y,z]), del_tri.simplices

## Disk 
nr, nc = 16, 24 # N = nr + np + 1 
r, theta = np.linspace(1.0/(nr-1), 1.0, nr), np.linspace(0, 2*np.pi, nc)
r, theta = np.meshgrid(r, theta)
r, theta = r.flatten(), theta.flatten()
x, y = np.append(r*np.cos(theta), 0), np.append(r*np.sin(theta), 0) # polar -> cartesian,  add center
z = np.sin(-x*y)
del_tri = Delaunay(np.c_[x,y])
V, T = scale_diameter(np.c_[x,y,z]), del_tri.simplices

## Project onto vector + show inner product
project_v = lambda X, v: (X @ np.array(v)[:,np.newaxis]).flatten()
plot_direction(V, T, project_v(V, [0, 0, 1]))

## Archimedian spiral 
import matplotlib
S2_vp = archimedean_sphere(n=200, nr=15)
colors = np.array([matplotlib.colors.to_hex(cm.jet(v)) for v in np.linspace(0.0, 1.0, S2_vp.shape[0])])
fig = plt.figure(figsize=(15, 15))
ax = visualization.plot(S2_vp, space="S2", color=colors, alpha=0.7)
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])


D0, D1, D2, D2_h, D2_g = lower_star_pb_terms(V, T, project_v(V, [0,0,1]), a=0.21, b=0.40)

np.sum(np.linalg.eigh(D)[0])
np.sum(np.array([np.sum(np.diag(D)[:(96-i)])-np.sum(np.diag(D)[:(95-i)]) for i in range(96)]))


t1_1 = np.sum(np.linalg.svd(D0.A/0.21, compute_uv=False))
t2_1 = np.sum(np.linalg.svd(D1.A/0.21, compute_uv=False))
t3_1 = np.sum(np.linalg.svd(D2.A/0.40, compute_uv=False))
t4_1 = np.sum(np.linalg.svd((D2_h.multiply(D2_g)).A/0.40, compute_uv=False))

D0_2, D1_2, D2_2, D2_h_2, D2_g_2 = lower_star_pb_terms(V, T, project_v(V, [0,0,1]), a=0.22, b=0.40)

t1_2 = np.sum(np.linalg.svd(D0_2.A/0.22, compute_uv=False))
t2_2 = np.sum(np.linalg.svd(D1_2.A/0.22, compute_uv=False))
t3_2 = np.sum(np.linalg.svd(D2_2.A/0.40, compute_uv=False))
t4_2 = np.sum(np.linalg.svd((D2_h_2.multiply(D2_g_2)).A/0.40, compute_uv=False))

(t1_1 - t2_1 - t3_1 + t4_1) <= (t1_2 - t2_2 - t3_2 + t4_2)

r = np.zeros((100, 4))
for i, aa in enumerate(np.linspace(0.05, 0.35, 100)):
  D0_2, D1_2, D2_2, D2_h_2, D2_g_2 = lower_star_pb_terms(V, T, project_v(V, [0,0,1]), a=aa, b=0.40)
  s1 = np.linalg.svd(D0_2.A/aa, compute_uv=False)
  s2 = np.linalg.svd(D1_2.A/aa, compute_uv=False)
  s3 = np.linalg.svd(D2_2.A/0.40, compute_uv=False)
  s4 = np.linalg.svd((D2_h_2.multiply(D2_g_2)).A/0.40, compute_uv=False)
  eps = 0.001
  t1_2 = np.sum((s1**2)/(s1**2 + eps))
  t2_2 = np.sum((s2**2)/(s2**2 + eps))
  t3_2 = np.sum((s3**2)/(s3**2 + eps))
  t4_2 = np.sum((s4**2)/(s4**2 + eps))
  # t1_2 = np.sum(np.linalg.svd(D0_2.A, compute_uv=False))
  # t2_2 = np.sum(np.linalg.svd(D1_2.A, compute_uv=False))
  # t3_2 = np.sum(np.linalg.svd(D2_2.A, compute_uv=False))
  # t4_2 = np.sum(np.linalg.svd((D2_h_2.multiply(D2_g_2)).A, compute_uv=False))
  r[i,:] = [t1_2, t2_2, t3_2, t4_2]
  # r[i] = lower_star_pb(V, T, project_v(V, [0,0,1]), a=aa, b=0.40, type="rank")

plt.plot(r)


r = np.zeros((100, 4))
a,b = 0.21, 0.40
for a in np.linspace(0.20, 0.395, 8):
  for i, eps in enumerate(np.linspace(1e-12, 1e-3, 40)):
    D0_2, D1_2, D2_2, D2_h_2, D2_g_2 = lower_star_pb_terms(V, T, project_v(V, [0,0,1]), a=a, b=b)
    s1 = np.linalg.svd(D0_2.A/a, compute_uv=False)
    s2 = np.linalg.svd(D1_2.A/a, compute_uv=False)
    s3 = np.linalg.svd(D2_2.A/b, compute_uv=False)
    s4 = np.linalg.svd((D2_h_2.multiply(D2_g_2)).A/b, compute_uv=False)
    t1_2 = np.sum((s1**2)/(s1**2 + eps))
    t2_2 = np.sum((s2**2)/(s2**2 + eps))
    t3_2 = np.sum((s3**2)/(s3**2 + eps))
    t4_2 = np.sum((s4**2)/(s4**2 + eps))
    # t1_2 = np.sum(np.linalg.svd(D0_2.A, compute_uv=False))
    # t2_2 = np.sum(np.linalg.svd(D1_2.A, compute_uv=False))
    # t3_2 = np.sum(np.linalg.svd(D2_2.A, compute_uv=False))
    # t4_2 = np.sum(np.linalg.svd((D2_h_2.multiply(D2_g_2)).A, compute_uv=False))
    r[i,:] = [t1_2, t2_2, t3_2, t4_2]
  # r[i] = lower_star_pb(V, T, project_v(V, [0,0,1]), a=aa, b=0.40, type="rank")
  plt.plot(r[:,0] - r[:,1] - r[:,2] + r[:,3])
A = np.linspace(0.05, 0.35, 100)

np.sum(np.diag(D0.A))
np.sum(2*(np.diag(D0.A)**2))

np.diag(D1.A.T @ D1.A) - 2*(np.diag(D0.A)**2) ## all close! 

(-np.sort(-np.diag(D1.A.T @ D1.A)))[:5] # sum of squared singular values 

np.linalg.svd(D1.A, compute_uv=False)[:5]

np.sum(np.sqrt(np.diag(D1.A.T @ D1.A)))


disk_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40) for v in S2_vp])

mb_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40) for v in S2_vp])

tor_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40) for v in S2_vp])


fig = plt.figure(figsize=(12,18), dpi=350)
ax = plt.gca()
ax.plot(mb_curve, c='green', label='Mobius Band')

fig, axs = plt.subplots(1, 3, figsize=(18, 3))
axs[0].plot(disk_curve, c='blue', label='Disk')
axs[1].plot(mb_curve, c='green', label='Mobius Band')
axs[2].plot(tor_curve, c='orange', label='Torus')






tor_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='fro') for v in S2_vp])
mb_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='fro') for v in S2_vp])
disk_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='fro') for v in S2_vp])

fig, axs = plt.subplots(1, 3, figsize=(18, 3))
axs[0].plot(disk_curve, c='blue', label='Disk')
axs[1].plot(mb_curve, c='green', label='Mobius Band')
axs[2].plot(tor_curve, c='orange', label='Torus')


tor_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='approx') for v in S2_vp])
mb_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='approx') for v in S2_vp])
disk_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='approx') for v in S2_vp])


tor_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='rank') for v in S2_vp])
mb_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='rank') for v in S2_vp])
disk_curve = np.array([lower_star_pb(V, T, project_v(V, v), a=0.20, b=0.40, type='rank') for v in S2_vp])

import matplotlib
cmap = matplotlib.cm.get_cmap('jet')
colors = list(np.array([cmap(v) for v in np.linspace(0, 1, endpoint=True)]))
face_colors = bin_color(TW, colors)

v = np.array([0.0, 0.0, 1.0])
W = (V @ v[:,np.newaxis]).flatten()
TW = np.array([np.max(W[t]) for t in T])

import functools
def radius(R: float = 1.0):
  def dec_func(func): 
    @functools.wraps(func)
    def radius_dec(*args, **kwargs):
      return(R*func(*args, **kwargs))
    return(radius_dec)

@radius(5)
def circle_family(n: int): 
  angle = np.linspace(0, 2*np.pi, n, endpoint=False)
  return(np.c_[np.cos(angle), np.sin(angle)])

dom = np.linspace(0, 5*(2*np.pi), 1200)
plt.plot(dom, np.cos(dom+np.pi))
np.correlate(np.cos(dom), np.cos(dom+np.pi))


import matplotlib.pyplot as plt 
plt.scatter(*circle_family(10).T)

E = edges_from_triangles(T, V.shape[0])

## First, translate and rescale to have diameter 1
V = scale_diameter(V - V.mean(axis=0))

v_weights = V @ v[:,np.newaxis]
e_weights = np.maximum(v_weights[E[:,0]], v_weights[E[:,1]]).flatten()
t_weights = np.maximum(v_weights[T[:,0]], v_weights[T[:,1]], v_weights[T[:,2]]).flatten()

rank_C2(*T[0,[0,1]], n=V.shape[0])


v[:,np.newaxis]









from tallem.datasets import scatter3D
%matplotlib
scatter3D(V)
%matplotlib inline


from pyqtgraph.opengl import GLViewWidget, MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
mesh = MeshData(V / 100, T) 
item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")
view = GLViewWidget()
view.addItem(item)
view.show()


from scipy.spatial.distance import cdist 
from dms import circle_family
from tallem.dimred import cmds
C, _ = circle_family(n=32, sd=0.03)
W0 = np.ones(shape=(32,32)) #np.random.uniform(size=(32,32))
W1 = np.random.uniform(size=(32,32))
W1 = W1 @ W1.T
W1 = (10.0) + (W1 / np.max(W1))
W = lambda t: (1-t)*W0 + t*W1
D = cdist(C(0.05), C(0.05))

np.all(np.argsort((D * W(0.0)).flatten()) == np.argsort((D * W(0.1)).flatten()))

import matplotlib.pyplot as plt
plt.scatter(*cmds(D*W(0)).T)
plt.scatter(*cmds(D*W(1.0)).T)


dom = np.linspace(1e-6, 5, 1000)
f = lambda x: 15 + np.log(3*x)
g = lambda x: -(15 + np.log(2*x))
plt.plot(dom, f(dom))
plt.plot(dom, g(dom))



## Basic shapes via implicit equations
import mcubes
X, Y, Z = np.mgrid[:50, :50, :50]
X, Y, Z = X-25, Y-25, Z-25

X, Y, Z = np.mgrid[range(-30, 30, 51), range(-30, 30, 51), range(-30, 30, 51)]

## Lemniscate of Bernoulli:
U = ((X**2 + Y**2)**2 - X**2 + Y**2)**2 + Z**2
V, T = mcubes.marching_cubes(U, 0.05)

## Torus 
U = (np.sqrt(X**2 + Y**2) - 5.0)**2 + Z**2 - 2.0**2 ## R, r
V, T = mcubes.marching_cubes(U, 0.0)

# R = 5.0
# U = -R**2 * Y  + X**2 * Y + Y**3 - 2*R*X*Z - 2*X**2 * Z - 2 * Y**2 * Z + Y * Z**2
# V, T = mcubes.marching_cubes(U, 0.0)



import os.path
from pathlib import Path
import pywavefront

elephant_path = os.path.expanduser('~/Downloads') + '/elephant-poses/elephant-reference.obj'
elephant_path = Path(elephant_path).resolve()
shape = pywavefront.Wavefront(str(elephant_path), collect_faces=True)

# from pywavefront import visualization
# visualization.draw(shape)

V, T = np.array(shape.vertices), np.array(shape.mesh_list[0].faces, dtype=np.int32)

import plotly.graph_objects as go
axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
layout = go.Layout(scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis), aspectratio=dict(x=1, y=1, z=1)))
fig = go.Figure(data=[go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=T[:,0],j=T[:,1],k=T[:,2])], layout=layout)
fig.show()





T = np.array([[0,2,3], [1,2,3], [0,1,2], [1,2,4], [0,3,4]])
E = edges_from_triangles(T, 5)
W = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ER = np.sort(rank_combs(E, k=2, n=len(W)))
TR = np.sort(rank_combs(T, k=3, n=len(W)))

# unrank_comb(2, k=3, n=4)
# rank_combs([[0,2], [0,3], [2,3]], k=2, n=4)

from scipy.sparse import csc_matrix
from apparent_pairs import rank_combs

T.sort(axis=1)
T = T[np.lexsort(np.rot90(T))]
TR = np.sort(rank_combs(T, k=3, n=len(V)))
#NT2 = unrank_combs(TR, 3, len(V))

E = edges_from_triangles(T, len(V))
ER = np.sort(rank_combs(E, k=2, n=len(V)))
TR = np.sort(rank_combs(T, k=3, n=len(V)))
v = np.array([0, 0, 1.0])
W = (V @ v[:,np.newaxis]).flatten()

cdata, cindices, cindptr, ew = boundary.lower_star_boundary_1_sparse(W, ER)
D1 = csc_matrix((cdata, cindices, cindptr), shape=(len(W), len(cindptr)-1))

cdata, cindices, cindptr, tw = boundary.lower_star_boundary_2_sparse(W, ER, TR)
D2 = csc_matrix((cdata, cindices, cindptr), shape=(len(ER), len(cindptr)-1))


## Form the persistent Betti relaxation for a sparse lower star 
# E = edges_from_triangles(T, nv=V.shape[0])


D1, ew = lower_star_boundary(W, simplices=E)
D2, tw = lower_star_boundary(W, simplices=T)

v = np.array([1.0, 0, 0])
W = (V @ v[:,np.newaxis]).flatten()
W += abs(np.min(W)) + 0.10





from ripser import ripser
from tallem.distance import as_dist_matrix
from itertools import combinations



D = np.zeros(shape=(V.shape[0], V.shape[0]))
E = edges_from_triangles(T, len(W))
ER = np.sort(rank_combs(E, 2, len(W)))
for (i,j) in combinations(range(V.shape[0]), 2):
  if rank_C2(i,j,len(W)) in ER:
    D[i,j] = D[j,i] = np.max([W[i], W[j]])



# d = np.zeros(comb(len(V), 2))
# d[rank_combs(E, 2, len(V))] = ew
# D = as_dist_matrix(d)
D += W*np.eye(D.shape[0])
Ds = csc_matrix(D)
Ds.eliminate_zeros()
dgms = ripser(Ds, distance_matrix=True, maxdim=1)

dgms['dgms'][1]

## annulus 
theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
circle = np.c_[np.cos(theta), np.sin(theta)]

p = (2*np.pi)/(24*2)
R = np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
outer = (R @ (1.5*circle).T).T
X = np.vstack((circle, outer))
plt.scatter(*X.T)
plt.gca().set_aspect('equal')

Z = np.vstack((X, [0.0, 0.0]))
DT = Delaunay(Z)
S = np.array(list(filter(lambda s: not((Z.shape[0]-1) in s), DT.simplices)))

plt.scatter(*X.T)
plt.gca().set_aspect('equal')
for s in S:
  ind = np.append(s, s[0])
  plt.plot(X[ind, 0], X[ind, 1])

T = S
V = X
W = V[:,0]
E = edges_from_triangles(T, V.shape[0])

plt.scatter(*X.T, c=W)
plt.gca().set_aspect('equal')
for s in S:
  ind = np.append(s, s[0])
  plt.plot(X[ind, 0], X[ind, 1])

## Compare with dionysus 
dgms = lower_star_ph_dionysus(V, W, T)
for i, dgm in enumerate(dgms):
  for pt in dgm:
    print(i, pt.birth, pt.death)

def lower_star_ph_dionysus(V, W, T):
  from betti import edges_from_triangles
  E = edges_from_triangles(T, V.shape[0])
  vertices = [([i], w) for i,w in enumerate(W)]
  edges = [(list(e), np.max(W[e])) for e in E]
  triangles = [(list(t), np.max(W[t])) for t in T]
  F = []
  for v in vertices: F.append(v)
  for e in edges: F.append(e) 
  for t in triangles: F.append(t)

  import dionysus as d
  f = d.Filtration()
  for vertices, time in F: f.append(d.Simplex(vertices, time))
  f.sort()
  m = d.homology_persistence(f)
  dgms = d.init_diagrams(m, f)
  DGM0 = np.array([[pt.birth, pt.death] for pt in dgms[0]])
  DGM1 = np.array([[pt.birth, pt.death] for pt in dgms[1]])
  return([DGM0, DGM1])

def lower_star_ph_ripser(V: ArrayLike, W: ArrayLike, T: ArrayLike):
  from ripser import ripser
  assert len(W) == V.shape[0]
  D = np.zeros(shape=(V.shape[0], V.shape[0]))
  E = edges_from_triangles(T, len(W))
  ER = np.sort(rank_combs(E, 2, len(W)))
  for (i,j) in combinations(range(V.shape[0]), 2):
    if rank_C2(i,j,len(W)) in ER:
      D[i,j] = D[j,i] = np.max([W[i], W[j]])
  D += W*np.eye(D.shape[0])
  Ds = csc_matrix(D)
  Ds.eliminate_zeros()
  dgms = ripser(Ds, distance_matrix=True, maxdim=1)  
  return(dgms)

lower_star_ph_dionysus(V, W, T)
lower_star_ph_ripser(V, W, T)



lower_star_pb(V, T, W, a=0.40, b=0.50)