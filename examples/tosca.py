import numpy as np
from numpy.typing import ArrayLike 
from typing import *
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import re
import requests

# import lzma
# import requests
# url = "https://raw.githubusercontent.com/peekxc/pbsig/main/src/pbsig/data/tosca/tosca.tar.xz"
# r = requests.get(url, stream=True)
# z = lzma.decompress(r.content)

# lzma.open(z)

import requests
import tarfile
url = "https://raw.githubusercontent.com/peekxc/tosca_signatures/main/tosca.tar.xz"
response = requests.get(url, stream=False)
tosca_file = tarfile.open(fileobj=response.raw, mode="r:xz")
with tarfile.open(fileobj=response.raw, mode="r:xz") as tosca_file:
  tosca_file.extractall(path=".") 
  for member in tosca_file.getmembers():
    if member.name[0] != "." and member.size > 0:
      print(member.name)
      scipy.io.loadmat(tosca_file.extractfile(member))


import urllib.request
import tarfile
filename = url.split("/")[-1]
urllib.request.urlretrieve(url, filename)
tosca_file = tarfile.open(filename, "r:xz")
wut = []
for member in tosca_file.getmembers():
  f = tosca_file.extractfile(member)
  print(f.name)
  wut.append(f)
  if f.name is not None:
    S = scipy.io.loadmat(f.read())
    print(type(S))

for tf in tosca_file:
  if tf.name[0] != ".":
    scipy.io.loadmat(tosca_file.extractfile(tf).read())

with open("filename", 'wb') as f:
  f.write(r.content)

# %% Tosca preprocessing 
tosca_dir = "/Users/mpiekenbrock/pbsig/src/pbsig/data/tosca/"
mat_rgx, cls_rgx = re.compile(r".*[.]mat"), re.compile(r"^([a-zA-Z]+)(\d+)[.]mat")
tosca_files = [fn for fn in os.listdir(tosca_dir) if mat_rgx.match(fn) is not None]

tosca_classes = ['dog', 'cat', 'michael', 'centaur', 'victoria', 'horse', 'david', 'gorilla', 'wolf'] 
tosca_colors = ['red', 'blue', 'black', 'orange', 'yellow', 'green', 'purple', 'pink', 'cyan']
tosca = { cls_nm : [] for cls_nm in  tosca_classes}
for fn in tosca_files:
  cls_nm, obj_id = cls_rgx.split(fn)[1:3] 
  tosca[cls_nm].append(int(obj_id))

def tosca_model(file_path: str, normalize: bool = True):
  mat = scipy.io.loadmat(file_path)
  x = np.ravel(mat['surface']['X'][0][0]).astype(float)
  y = np.ravel(mat['surface']['Y'][0][0]).astype(float)
  z = np.ravel(mat['surface']['Z'][0][0]).astype(float)
  S = np.c_[x,y,z]
  T = mat['surface']['TRIV'][0][0] - 1 # TOSCA is 1-based 
  if normalize:
    S -= S.mean(axis=0)
    c = np.linalg.norm(S.min(axis=0) - S.max(axis=0))
    S *= (1/c)
  return S, T


# %% Plot random samples
from itertools import product
nrow, ncol = 4, 5
zoom = 0.33
tosca_models = np.random.choice(tosca_files, size=nrow*ncol, replace=False)

fig, axs = plt.subplots(nrow, ncol, figsize=(150, 150), subplot_kw={'projection': '3d'})
idx_iter = product(range(nrow), range(ncol))
for model_name, (i,j) in zip(tosca_models, idx_iter):
  S, T = tosca_model(tosca_dir + model_name)
  c, rng = S.mean(axis=0), max(abs(S.max(axis=0)-S.min(axis=0)))
  axs[i,j].scatter(*S.T, s=15.48, c=S[:,2])
  axs[i,j].set_xlim(c[0] - zoom*rng, c[0] + zoom*rng)
  axs[i,j].set_ylim(c[1] - zoom*rng, c[1] + zoom*rng)
  axs[i,j].set_zlim(c[2] - zoom*rng, c[2] + zoom*rng)
  axs[i,j].axis('off')


def euler_curve(X: ArrayLike, T: ArrayLike, f: ArrayLike, bins: Union[int, Sequence[float]] = 20, method: str = ["simple", "top_down"]) -> ArrayLike:
  """Calculates the euler characteristic curve.

  Parameters:
    X = (n x 3) array of vertex coordinates 
    T = (m x 3) array of triangle indices 
    f = array of vertex values to filter by 
    bins = number of bins, or a sequence of threshold values 
    method = strng indicating which method to use to compute the curve. Defaults to "simple". 

  Returns: 
    array of euler values of the mesh along _bins_ 
  """
  assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 3, "Invalid point cloud given."
  assert isinstance(T, np.ndarray) and T.ndim == 2 and T.shape[1] == 3, "Invalid triangles given."
  f = np.array(f)
  assert isinstance(f, np.ndarray) and len(f) == X.shape[0], "Invalid vertex function array given."
  bins = np.linspace(min(f), max(f)+10*np.finfo(float).eps, bins) if isinstance(bins, int) else bins
  assert isinstance(bins, Collection), f"Invalid argument bins={bins}"
  
  ## Choose the euler invariant 
  if isinstance(method, list) or method == "simple" or method is None:
    ## Vectorized O(nk) approach
    E = np.array(list(set([tuple(e) for e in T[:,[0,1]]]) | set([tuple(e) for e in T[:,[0,2]]]) | set([tuple(e) for e in T[:,[1,2]]])))
    fe = f[E].max(axis=1)
    ft = f[T].max(axis=1)
    ecc = np.array([sum(f < t) - sum(fe < t) + sum(ft < t) for t in bins])
  else: 
    ## O(n + k) approach from "Efficient classification using the Euler characteristic"
    from itertools import combinations
    vw, ew = {}, {}
    ft = f[T].max(axis=1)
    for t, coface_value in zip(T, ft):
      for e in combinations(t,2):
        ew[e] = max(ew[e], coface_value) if e in ew else coface_value
      for v in combinations(t,1):
        vw[v] = max(vw[v], coface_value) if v in vw else coface_value
    v_counts = np.cumsum(np.histogram(list(vw.values()), bins=bins)[0])
    e_counts = np.cumsum(np.histogram(list(ew.values()), bins=bins)[0])
    t_counts = np.cumsum(np.histogram(ft, bins=bins)[0])
    ecc = np.append(0, (v_counts - e_counts + t_counts))
  return ecc


# %% 
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
# tosca_subset = np.random.choice(tosca_classes, size = 4, replace=False)
tosca_subset = ["dog", "cat", "horse", "centaur", "victoria"]
bins = 20
INDEX = np.arange(bins)

p = figure(width=450, height=250, x_axis_label="Height index", y_axis_label="Euler Characteristic")
for tc in tosca_subset:
  for obj_id in tosca[tc]:
    X, T = tosca_model(tosca_dir + tc + str(obj_id) + ".mat")
    f = X @ np.array([0,0,1])
    ecc = euler_curve(X, T, f, method="simple")
    p.line(INDEX, ecc, color=tosca_colors[tosca_classes.index(tc)], legend_label=tc)
p.legend.location = "bottom_left"
show(p)

# %% Curvature 
import pymeshlab
def curvature(
  X: ArrayLike, 
  T: ArrayLike, 
  value=["mean", "gaussian", "min", "max", "shape_index", "curvedness"], 
  fit=['Quadric Fitting', 'Normal Cycles', 'Taubin approximation']
) -> ArrayLike:
  """uses pymeshlab to compute the principal directions of curvature with different algorithms. 
  
  Parameters: 
    X = (n x 3) array of vertex coordinates 
    T = (m x 3) array of triangle indices 
    value = type of curvature to compute. 
    fit = algorithm to fit the curvature by. 

  Returns: 
    array of curvature values at the mesh vertices.

  Note: This function can fail if the mesh has non-unique vertices

  See: https://pymeshlab.readthedocs.io/en/2021.10/filter_list.html#compute_curvature_principal_directions for more details. 
  """ 
  fit_method = {'quadric fitting' : 3, 'normal cycles' : 2, 'taubin approximation' : 0 }
  value_type = {'mean': 0, 'gauss': 1, 'min': 2, 'max': 3, 'shape_index': 4, 'curvedness': 5 }
  value = 'shape_index' if isinstance(value, list) else str(value).tolower()
  fit = 'quadric fitting' if isinstance(fit, list) else str(fit).tolower()
  assert isinstance(fit, str) and fit in fit_method.keys(), f"Invalid fit method '{fit}' given."
  assert isinstance(value, str) and value in value_type.keys(), f"Invalid curvature type '{value}' given."
  ms = pymeshlab.MeshSet()
  ms.add_mesh(pymeshlab.Mesh(X, T))
  ms.compute_curvature_principal_directions_per_vertex(method=3, curvcolormethod=4)
  mesh = ms.mesh(0)
  return mesh.vertex_scalar_array()

# %% Euler characteristic with varying curvature
p = figure(width=450, height=250, x_axis_label="Sublevel sets of Shape Index", y_axis_label="Euler Characteristic")
for tc in tosca_subset:
  for obj_id in tosca[tc]:
    X, T = tosca_model(tosca_dir + tc + str(obj_id) + ".mat")
    f = curvature(X, T)
    ecc = euler_curve(X, T, f, method="simple")
    p.line(INDEX, ecc, color=tosca_colors[tosca_classes.index(tc)], legend_label=tc)
p.legend.location = "bottom_left"
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

# time_pts = np.linspace(min(fv), p*max(fv), k)
# time_pts[-1]
# offset = (max(ft) - min(ft))
# buckets = np.zeros(k, dtype=int)
# # indices = np.minimum(np.floor((k * ft) / time_pts[-1]).astype(np.int16), k-1)
# indices = np.minimum(np.floor((ft + offset)*k / (max(ft) + offset)).astype(np.int16), k-1)
# np.histogram(ft, bins=k)
# np.add.at(buckets, indices, 1)  #scale=pymeshlab.Percentage(0.5)

