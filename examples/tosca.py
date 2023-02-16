import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook

tosca_dir = "/Users/mpiekenbrock/pbsig/src/pbsig/data/tosca/"
tosca_files = os.listdir(tosca_dir)

mat = scipy.io.loadmat('/Users/mpiekenbrock/pbsig/src/pbsig/data/tosca/dog3.mat')
x = np.ravel(mat['surface']['X'][0][0]).astype(float)
y = np.ravel(mat['surface']['Y'][0][0]).astype(float)
z = np.ravel(mat['surface']['Z'][0][0]).astype(float)
S = np.c_[x,y,z]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*S.T, s=0.18, c=S[:,2])
c = S.mean(axis=0)
rng = max(abs(S.max(axis=0)-S.min(axis=0)))
zoom = 0.33
ax.set_xlim(c[0] - zoom*rng, c[0] + zoom*rng)
ax.set_ylim(c[1] - zoom*rng, c[1] + zoom*rng)
ax.set_zlim(c[2] - zoom*rng, c[2] + zoom*rng)

T = mat['surface']['TRIV'][0][0]
E = np.array(list(set([tuple(e) for e in T[:,[0,1]]]) | set([tuple(e) for e in T[:,[0,2]]]) | set([tuple(e) for e in T[:,[1,2]]])))
fv = f(S, T)
fe = fv[E-1].max(axis=1)
ft = fv[T-1].max(axis=1)


## Euler characteristic curve
def ecc_tosca(file_path, f: Callable, k: int = 10, p: float = 1.0):
  """Calculates euler characteristic curve on TOSCA data set.
  
  k := number of buckets
  p := percentage of maximum 
  Requires scalar-valued function f(X) to filter the sublevel sets of a mesh whose vertices are embedded in 'X'
  """
  mat = scipy.io.loadmat(file_path)
  x = np.ravel(mat['surface']['X'][0][0]).astype(float)
  y = np.ravel(mat['surface']['Y'][0][0]).astype(float)
  z = np.ravel(mat['surface']['Z'][0][0]).astype(float)
  S = np.c_[x,y,z]
  S -= S.mean(axis=0)
  # norms = np.linalg.norm(S, axis=1)
  c = np.linalg.norm(S.min(axis=0) - S.max(axis=0))
  # S = np.array([c*(s/n) for s,n in zip(S,   (1/c)*norms)])
  S *= (1/c)
  T = mat['surface']['TRIV'][0][0]
  E = np.array(list(set([tuple(e) for e in T[:,[0,1]]]) | set([tuple(e) for e in T[:,[0,2]]]) | set([tuple(e) for e in T[:,[1,2]]])))
  
  fv = f(S, T)
  fe = fv[E-1].max(axis=1)
  ft = fv[T-1].max(axis=1)
  time_pts = np.linspace(min(fv), p*max(fv), k)
  time_pts[-1]
  offset = (max(ft) - min(ft))
  buckets = np.zeros(k, dtype=int)
  # indices = np.minimum(np.floor((k * ft) / time_pts[-1]).astype(np.int16), k-1)
  indices = np.minimum(np.floor((ft + offset)*k / (max(ft) + offset)).astype(np.int16), k-1)
  np.histogram(ft, bins=k)
  np.add.at(buckets, indices, 1)

  fe_max = np.zeros(E.shape[0])
  ecc = [sum(fv < t) - sum(fe < t) + sum(ft < t) for t in np.linspace(min(fv), p*max(fv), N)]
  return ecc

## Choose a class to render 
dogs = [fn for fn in tosca_files if fn[:3] == "dog" and fn[-3:] == "mat"]
cats = [fn for fn in tosca_files if fn[:3] == "cat" and fn[-3:] == "mat"]
horses = [fn for fn in tosca_files if fn[:5] == "horse" and fn[-3:] == "mat"]

## Filter sublevel sets via simple height function / the z-coordinate
f = lambda X, T: X @ np.array([0,0,1])

from bokeh.plotting import figure, show
N = 20
INDEX = np.arange(20)
p = figure(width=450, height=250)
dog_r = [p.line(INDEX, ecc_tosca(tosca_dir + dog, f, N), color='blue') for dog in dogs]
cat_r = [p.line(INDEX, ecc_tosca(tosca_dir + cat, f, N), color='black') for cat in cats]
horses_r = [p.line(INDEX, ecc_tosca(tosca_dir + horse, f, N), color='green') for horse in horses]
show(p)


import pymeshlab

def curvature(X, T, ):
  ms = pymeshlab.MeshSet()
  ms.add_mesh(pymeshlab.Mesh(X, T))
  # assert mesh.is_compact()
  ## Uses the normal cycles (2) to compute shape index (4)
  ms.compute_curvature_principal_directions_per_vertex(method=3, curvcolormethod=4)
  #scale=pymeshlab.Percentage(0.5)
  mesh = ms.mesh(0)
  return mesh.vertex_scalar_array()

f = lambda X,T: curvature(X, T-1)

from bokeh.plotting import figure, show
N = 20
INDEX = np.arange(20)
p = figure(width=450, height=250)
dog_r = [p.line(INDEX, ecc_tosca(tosca_dir + dog, f, N, 0.10), color='blue') for dog in dogs]
cat_r = [p.line(INDEX, ecc_tosca(tosca_dir + cat, f, N,  0.10), color='black') for cat in cats]
horses_r = [p.line(INDEX, ecc_tosca(tosca_dir + horse, f, N,  0.10), color='green') for horse in horses]
show(p)



# import autograd.numpy as np
# from autograd import elementwise_grad, jacobian, hessian
# def shape_index(vertices, triangles, radius=0.1):
#     """
#     Calculates the shape index of a surface triangulation using arc tangents and eigenvalues of the Hessian matrix.
    
#     Parameters:
#         vertices (np.ndarray): An array of shape (n, 3) containing the (x, y, z) coordinates of n vertices
#         triangles (np.ndarray): An array of shape (m, 3) containing the vertex indices of m triangles
#         radius (float): The radius to use for the curvature estimation (default: 0.1)
    
#     Returns:
#         shape_idx (np.ndarray): Shape index for each vertex in the mesh
#     """
#     def curvature(x, y, z): # Gaussian curvatures
#       J = jacobian(lambda p: np.array([x(p), y(p), z(p)]))
#       H = hessian(lambda p: np.array([x(p), y(p), z(p)]))
#       g = np.sum(J * J, axis=1)
#       H = H.reshape((3, 3, -1))
#       c = np.zeros_like(g)
#       for i in range(len(c)):
#         H_i = H[:, :, i]
#         c[i] = np.linalg.det(H_i) / (g[i]**3)
#       return c
#     curvatures = curvature(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    
#     hessians = np.zeros((vertices.shape[0], 3, 3))
#     for i in range(vertices.shape[0]):
#       x = elementwise_grad(lambda p: curvature(p[:, 0], p[:, 1], p[:, 2]))(vertices[i:i+1, :] + radius * np.random.randn(1, 3))
#       hessians[i, :, :] = 2 * np.mean(x, axis=0)
#     shape_idx = np.zeros(vertices.shape[0])
    
#     for i in range(vertices.shape[0]):
#       H = hessians[i, :, :]
#       eigenvalues = np.linalg.eigvals(H)
#       shape_idx[i] = np.arctan2(eigenvalues[0] + eigenvalues[1], np.sqrt((eigenvalues[0] - eigenvalues[1])**2 + 4*eigenvalues[2]**2))
#     return shape_idx

# ## Generated via ChatGPT on 02/15/2023
# def shape_index(tri_mesh):    
#   # trimesh.curvature.discrete_mean_curvature_measure(tri_mesh, points=tri_mesh.vertices, radius=0.1)
#   hessians = trimesh.curvature.discrete_gaussian_curvature_measure(tri_mesh, points=tri_mesh.vertices, radius=0.1)
#   # hessians = trimesh.curvature.discrete_gaussian_curvature_measure(tri_mesh, points=tri_mesh.vertices, radius=0.1)
#   shape_idx = np.zeros(tri_mesh.vertices.shape[0])
#   for i in range(tri_mesh.vertices.shape[0]):
#     H = hessians['hessian'][i]
#     eigenvalues = np.linalg.eigvals(H)
#     shape_idx[i] = np.arctan2(eigenvalues[0] + eigenvalues[1], np.sqrt((eigenvalues[0] - eigenvalues[1])**2 + 4*eigenvalues[2]**2))
#   return shape_idx
# shape_index(mesh)
# ER = np.array([[rank_C2(*t[[0,1]], n=nv), rank_C2(*t[[0,2]], n=nv), rank_C2(*t[[1,2]], n=nv)] for t in triangles])
#   ER = np.unique(ER.flatten())
#   E = np.array([unrank_C2(r, n=nv) for r in ER])
# nv = len(mat['surface']['X'][0][0])
## Filter using a more geometric measure
# from trimesh import Trimesh, curvature
# def gauss_curv(X, T, r: float = 0.1):
#   mesh = Trimesh(vertices=X, faces=T)
#   gcurv = curvature.discrete_gaussian_curvature_measure(mesh, points=mesh.vertices, radius=r)
#   return gcurv
# f = lambda X,T: gauss_curv(X, T-1, 0.0001)