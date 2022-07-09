## lower_star boundary matrices 
import cppimport.import_hook
import boundary
import numpy as np

import geomstats

from matplotlib import cm

#archimedian_spiral over unit sphere
a_max = 15*(2*np.pi)
alpha = np.linspace(0, a_max, 1500)
# alpha = np.append(np.linspace(0, a_max, 1500), np.linspace(a_max, 0, 1500))
p = -np.pi/2 + (alpha*np.pi)/(a_max)
x = np.cos(alpha)*np.cos(p)
y = np.sin(alpha)*np.cos(p)
z = -np.sin(p)
X = np.c_[x,y,z]

p = -np.pi/2 + (alpha*np.pi)/(a_max)
x = -np.cos(alpha)*np.cos(p)
y = -np.sin(alpha)*np.cos(p)
z = -np.sin(p)
Y = np.c_[x,y,z]

Z = np.vstack((X, Y))

# v = np.array([1, 0, 0])[:,np.newaxis]
# R = np.eye(3) - 2*(v @ v.T)/(v.T @ v)
# Y = X @ R
# Z = np.vstack((X, X @ R))

import matplotlib.pyplot as plt
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)

fig = plt.figure(figsize=(15, 15))
ax = visualization.plot(Z, space="S2", color="red", alpha=0.7)
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])


from scipy.special import ellipeinc
from scipy.optimize import minimize

archi_arc_len = lambda alpha: ellipeinc((np.pi*alpha)/a_max, -(a_max**2) / (2*np.pi))
plt.plot(alpha, archi_arc_len(alpha))


max_len = archi_arc_len(np.max(alpha))

def archi_alpha(L: float) -> float:
  ''' Given length 'L', returns the corresponding alpha on the Archimedean spiral '''
  l2_error = lambda a, L: abs(archi_arc_len(a) - L)**2
  res = minimize(l2_error, 1.0, args=(L), method='Nelder-Mead', tol=1e-6)
  return(res.x[0])


def archimedean_sphere(n: int, nr: int):
  ''' Gives an n-point uniform sampling over 'nr' rotations around the 2-sphere '''
  from scipy.special import ellipeinc
  from scipy.optimize import minimize
  a_max = nr*(2*np.pi)
  alpha = np.linspace(0, a_max, n)
  archi_arc_len = lambda alpha: ellipeinc((np.pi*alpha)/a_max, -(a_max**2) / (2*np.pi))
  max_len = archi_arc_len(alpha[-1])
  alpha_equi = np.array([archi_alpha(L) for L in np.linspace(0.0, max_len, n)])
  p = -np.pi/2 + (alpha_equi*np.pi)/(a_max)
  x = np.cos(alpha_equi)*np.cos(p)
  y = np.sin(alpha_equi)*np.cos(p)
  z = -np.sin(p)
  X = np.vstack((np.c_[x,y,z], np.flipud(np.c_[-x,-y, z])))
  return(X)


import matplotlib
from matplotlib import cm
from tallem.color import bin_color
X = archimedean_sphere(1500, 4)

colors = np.array([matplotlib.colors.to_hex(cm.jet(v)) for v in np.linspace(0.0, 1.0, X.shape[0])])
fig = plt.figure(figsize=(15, 15))
ax = visualization.plot(X, space="S2", color=colors, alpha=0.7)
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])




#col_pal = np.array(['#%02x%02x%02x%02x' % tuple(int(i) for i in cm.jet(j)) for j in np.linspace(0.0, 1.0, len(x))])
#colors = bin_color(np.array(range(len(x)), dtype=float), color_pal=col_pal)

i = 80
v = x[i], y[i], z[i] # np.linalg.norm(v) == 1

def torus(n: int, inner_r: float, outer_r: float):
  theta = np.linspace(0, 2*np.pi, n)
  phi = np.linspace(0, 2*np.pi, n)
  theta, phi = np.meshgrid(theta, phi)
  theta, phi = theta.flatten(), phi.flatten()
  x = (outer_r + inner_r*np.cos(theta)) * np.cos(phi)
  y = (outer_r + inner_r*np.cos(theta)) * np.sin(phi)
  z = inner_r * np.sin(theta)
  return(np.c_[x,y,z])

from scipy.spatial.distance import pdist, cdist
T = torus(25, inner_r=0.5, outer_r=2.0)
T = T + np.random.uniform(size=T.shape, low=0.0, high=0.02)
T = T - T.mean(axis=0) # center to origin 

vec_mag = np.reshape(np.linalg.norm(T, axis=1), (T.shape[0], 1))
Tn = T / np.c_[vec_mag, vec_mag, vec_mag]
r = 0.5*np.max(pdist(T)) 
Ts = Tn * (np.c_[vec_mag, vec_mag, vec_mag]/r)


sphere = Hypersphere(dim=2)
fig = plt.figure(figsize=(15, 15))
ax = visualization.plot(np.array([[0.0, 0.0, 1.0]]), space="S2", alpha=0.0)
ax.scatter3D(*Ts.T)

# unit scale 
 # random noise

from tallem.dimred import cmds
Tn = cmds((pdist(T)/(2*np.max(pdist(T))))**2, 3)


scatter3D(T)

boundary.lower_star_boundary_1


def sample_torus(n: int, ar: float = 2.0, sd: float = 0):
  r = 1/ar
  from array import array
  x = array('d')
  while (len(x) < n):
    theta = np.random.uniform(size=n, low=0.0, high=2.0 * np.pi)
    jacobian_theta = (1 + r * np.cos(theta))/(2 * np.pi)
    density_threshold = np.random.uniform(size=n, low=0.0, high=1.0/np.pi)
    valid = jacobian_theta > density_threshold
    x.extend(theta[valid])
  theta = np.array(x)[:n]
  phi = np.random.uniform(size=n, low=0.0, high=2.0*np.pi)
  res = np.c_[
    (1.0 + r * np.cos(theta)) * np.cos(phi), 
    (1.0 + r * np.cos(theta)) * np.sin(phi), 
    r * np.sin(theta)
  ]
  return(res)

T = sample_torus(500)
T = T - T.mean(axis=0) # center to origin 
vec_mag = np.reshape(np.linalg.norm(T, axis=1), (T.shape[0], 1))
Tn = T / np.c_[vec_mag, vec_mag, vec_mag]
r = 0.5*np.max(pdist(T)) 
Ts = Tn * (np.c_[vec_mag, vec_mag, vec_mag]/r)

v = np.array([0.0, 0.0, 1.0])
W = Ts @ v[:,np.newaxis]

sphere = Hypersphere(dim=2)
fig = plt.figure(figsize=(15, 15))
ax = visualization.plot(np.array([[0.0, 0.0, 1.0]]), space="S2", alpha=0.0)
ax.scatter3D(*Ts.T, c=W)




_,_,_, sw = boundary.lower_star_boundary_1(W.flatten(), np.max(W))


from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

dgm = ripser(Ts)['dgms'][1]
plot_diagrams(dgm)

a,b = 0.20, 0.40

from itertools import combinations
from scipy.spatial import Delaunay
dt = Delaunay(Ts, qhull_options='QbB')


X, Y, Z = array('d'), array('d'), array('d')
for quad in dt.simplices:
  for tri in combinations(quad, 3):
    X.extend(np.append(Ts[tri,0], np.nan))
    Y.extend(np.append(Ts[tri,1], np.nan))
    Z.extend(np.append(Ts[tri,2], np.nan))

# %matplotlib
fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(np.array([[0.0, 0.0, 1.0]]), space="S2", alpha=0.0)
ax.scatter3D(*Ts.T, c=W)
# ax.plot3D(X,Y,Z, color='red', lw=0.01)
ax.plot_trisurf(dt, Z)

ax.plot_trisurf(Ts[:,0], Ts[:,1], Ts[:,2], cmap='viridis', edgecolor='none')
for i in range(25):
  for s in combinations(dt.simplices[i], 3):
    ax.plot_trisurf(Ts[s,0], Ts[s,1], Ts[s,2], cmap='viridis', edgecolor='black')





dt.simplices[0]
## Alex excersi