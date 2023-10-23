# %% Imports 
from itertools import *
from typing import * 
import networkx as nx
import numpy as np
import open3d as o3
from more_itertools import chunked, pairwise
from pbsig.betti import Sieve
from scipy.sparse.csgraph import floyd_warshall
from splex import *

# %% Bokeh configuration 
import bokeh
from bokeh.plotting import show, figure 
from bokeh.models import Range1d
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()

# %% 
from splex import *
from pbsig.datasets import pose_meshes
mesh_loader = pose_meshes(simplify=1000, which=["elephant"]) # 8k triangles takes ~ 40 seconds

def geodesics(mesh):
  from scipy.sparse.csgraph import floyd_warshall
  from scipy.spatial.distance import squareform, cdist
  import networkx as nx
  vertices = np.asarray(mesh.vertices)
  mesh.compute_adjacency_list()
  G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(mesh.adjacency_list) })
  A = nx.adjacency_matrix(G).tocoo()
  A.data = np.linalg.norm(vertices[A.row] - vertices[A.col], axis=1)
  #A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
  AG = floyd_warshall(A.tocsr())
  return AG

# %% Test code the box has the right multiplicity
from pbsig.betti import Sieve
from pbsig.linalg import spectral_rank
# elephant_mesh = mesh_loader[0]
# A = geodesics(elephant_mesh)
# mesh_ecc = A.max(axis=1)
# ecc_weight = lower_star_filter(mesh_ecc)
# S = simplicial_complex(np.asarray(elephant_mesh.triangles))
# sieve = Sieve(S, family=[ecc_weight], p = 1)
# sieve.pattern = np.array([[1.48, 1.68, 1.70, 1.88], [1.12, 1.28, 1.36, 1.56], [1.0, 1.2, 2.0, 2.2]])
# sieve.sift()
# sieve.summarize(spectral_rank)

# %% Generalize the build process
def mesh_loader(index: int):
  mesh_loader = pose_meshes(simplify=index, which=["elephant"]) # 8k triangles takes ~ 40 seconds
  elephant_mesh = mesh_loader[0]
  
  elephant_mesh.remove_degenerate_triangles()
  elephant_mesh.remove_duplicated_triangles()
  elephant_mesh.remove_duplicated_vertices()
  elephant_mesh.remove_unreferenced_vertices()
  # return elephant_mesh
  A = geodesics(elephant_mesh)
  mesh_ecc = A.max(axis=1)
  ecc_weight = lower_star_filter(mesh_ecc)
  S = simplicial_complex(np.asarray(elephant_mesh.triangles), "tree")
  return S, ecc_weight, elephant_mesh

## Create the Sequence container for the search
class SieveSparsifier(Sequence):
  def __init__(self, loader: Callable, lb: int, ub: int, pattern: np.ndarray):
    self.loader = loader
    self.lb = lb
    self.ub = ub 
    self.accesses = []
    self.sieve = None
    self.pattern = pattern

  def __getitem__(self, index: int):
    self.accesses.append(index)
    S, w, _ = self.loader(index)
    # mesh = self.loader(index)
    # print(f"Index: {index}, Complex: {len(np.asarray(mesh.triangles))}, ")
    print(f"Index: {index}, Complex: {str(S)}, ")
    self.sieve = Sieve(S, family=[w], p = 1)
    self.sieve.pattern = self.pattern
    self.sieve.sift()
    box_count = np.ravel(self.sieve.summarize(f=spectral_rank))
    return np.allclose(box_count, [4,2,1,2])

  def __len__(self) -> int: 
    return np.abs(self.ub - self.lb)

sparsifier = SieveSparsifier(
  loader = mesh_loader, 
  lb = 100, ub = 80000, 
  pattern=np.array([[1.48, 1.68, 1.70, 1.88], [1.12, 1.28, 1.36, 1.56], [1.0, 1.2, 2.0, 2.2], [1.44, 1.52, 1.54, 1.68]])
)
# sparsifier[100]


# %% Try open3d again 
import open3d as o3
from pbsig.color import bin_color
from trimesh.transformations import rotation_matrix

S, w, mesh = sparsifier.loader(800)
v_pos = np.asarray(mesh.vertices)
v_prin_dir = np.linalg.eigh(np.cov(v_pos.T))[1]

def normalize_direction(mesh):
  # v_pos = np.asarray(mesh.vertices)
  # v_prin_dir = np.linalg.eigh(np.cov(v_pos.T))[1]  
  camera_transformation = np.eye(4)
  camera_transformation[:3, :3] = v_prin_dir.T
  mesh.transform(camera_transformation)

## Configure the visualizer
vis = o3.visualization.Visualizer()
vis.create_window(window_name="Mesh Sparsification", width=640, height=640, visible=False)
opt = vis.get_render_option()
opt.background_color = [1, 1, 1]
opt.point_size = 5.0
opt.line_width = 1

## Get screenshots of the sparsified meshes
# indices = [100, 101, 103, 107, 115, 131, 163, 227, 355, 611, 1123, 867, 995, 994, 931, 963, 979, 987, 991, 993, 994, 993]
indices = [25,26,28,32,40,56,88,152,280,536,1048,2072,1560,1559,1304,1303,1176,1175,1112,1144,1160,1159,1152,1151,1148,1150,1149] # (25, 80k), k=1 exp search
for i, t_size in enumerate(indices):
  S, w, mesh = sparsifier.loader(t_size)
  mesh.compute_vertex_normals()
  #tri_ecc = w(np.array(mesh.triangles))
  #tri_colors = bin_color(tri_ecc, "turbo")

  normalize_direction(mesh)
  mesh.transform(rotation_matrix(0.35*np.pi, [0,1,0]))
  #mesh.vertex_colors = o3.utility.Vector3dVector(bin_color(w(faces(S,0)))[:,:3])

  vis.add_geometry(mesh)
  vis.poll_events() ## don't remove this
  vis.update_renderer()
  # vis.capture_depth_image("elephant_depth.png", False)
  vis.capture_screen_image(f"elephants_expsearch/elephant_{i}_{t_size:05d}.png", False)
  vis.remove_geometry(mesh)
  vis.poll_events() ## don't remove this
  vis.update_renderer()
  # vis.capture_screen_image(f"elephants/elephant_{i:05d}_removed.png", False)

mesh_loader = pose_meshes(simplify=None, which=["elephant"]) # 8k triangles takes ~ 40 seconds
elephant_mesh = mesh_loader[0]
elephant_mesh.remove_degenerate_triangles()
elephant_mesh.remove_duplicated_triangles()
elephant_mesh.remove_duplicated_vertices()
elephant_mesh.remove_unreferenced_vertices()
mesh = elephant_mesh

# %% Run exponential search to find the optimal sparsification
from pbsig import first_true_exp
first_true_exp(sparsifier, lb=25, ub=80000) ## [25, 80000] == 1150
sparsifier.accesses

# %% Log scale plot for the search
all_indices = np.array(indices + [80000])
markers = np.where(all_indices <= 1150, "x", "inverted_triangle")
markers[-1] = "square"
colors = np.where(all_indices <= 1150, "#ff0000a1", "#00ff00a1")
colors[-1] = "gray"
from bokeh.models import ColumnDataSource
scatter_data = ColumnDataSource({
  "x": all_indices, 
  "y": np.zeros(len(all_indices)),
  "marker" : markers, 
  "color":  colors 
})
p = figure(width=3920, height=int(0.6*100*(3920/1200)), x_axis_type="log", tools="") # x_axis_label="Sparsification Number of triangles"
p.output_backend = "svg"
p.scatter("x", "y", color="color", line_width=4, size=31.5, marker="marker", source=scatter_data)
p.yaxis.visible = False
# p.xaxis.axis_label = "Number of triangles"
p.xaxis.axis_line_width = 5
p.xaxis.minor_tick_out = 8
p.xaxis.minor_tick_line_width = 4.5
p.xaxis.major_tick_line_width = 6.5
p.xaxis.major_label_text_font_size = '48px'
p.toolbar_location = None
show(p)

from bokeh.io import export_png, export_svg
# plot.output_backend = "svg"
export_png(p, filename="elephants_expsearch/x_axis.png", width=3920, height=int(0.6*100*(3920/1200)))
# export_svg(p, filename="elephants_expsearch/x_axis.svg")

# %% Save the elephant images found during the exponential search
from pbsig.persistence import ph
from pbsig.vis import figure_dgm
from bokeh.models import Range1d

box_query = lambda dgm,i,j,k,l: np.sum( \
  (dgm[1]['birth'] >= i) & (dgm[1]['birth'] <= j) & \
  (dgm[1]['death'] >= k) & (dgm[1]['death'] <= l) \
)
## Compute ground-truth eccentricities for edges and vertices
## Final boxes: 
## card([1.48, 1.68] x [1.70, 1.88]) == 4   (legs)
## card([1.12, 1.28] x [1.36, 1.56]) == 2   (ears)
## card([1.0, 1.2] x [2.0, 2.2]) == 1       (torso)
## card([1.44, 1.52] x [1.54, 1.68]) == 2   (tusks)
## Rep: [25, 56, 536, 1149, 2072]
for index in [25, 56, 536, 1149, 2072]:
  mesh_loader = pose_meshes(simplify=index+1, which=["elephant"]) # 8k triangles takes ~ 40 seconds
  elephant_mesh = mesh_loader[0]
  elephant_mesh.remove_degenerate_triangles()
  elephant_mesh.remove_duplicated_triangles()
  elephant_mesh.remove_duplicated_vertices()
  elephant_mesh.remove_unreferenced_vertices()
  
  ## Compute the filtration + diagram
  A = geodesics(elephant_mesh)
  mesh_ecc = A.max(axis=1)
  ecc_weight = lower_star_filter(mesh_ecc)
  K_ecc = filtration(simplicial_complex(np.asarray(elephant_mesh.triangles)), ecc_weight)
  dgm = ph(K_ecc, engine="dionysus")

  ## Test constraints 
  c1 = "#00ff0051" if box_query(dgm, 1.48, 1.68, 1.70, 1.88) == 4 else "#ff000051"
  c2 = "#00ff0051" if box_query(dgm, 1.12, 1.28, 1.36, 1.56) == 2 else "#ff000051"
  c3 = "#00ff0051" if box_query(dgm, 1.00, 1.20, 2.00, 2.20) == 1 else "#ff000051"
  c4 = "#00ff0051" if box_query(dgm, 1.44, 1.52, 1.54, 1.68) == 2 else "#ff000051"
  reformat_box = lambda box: dict(x=box[0]+(box[1]-box[0])/2,y=box[2]+(box[3]-box[2])/2, width=box[1]-box[0],height=box[3]-box[2])

  p = figure_dgm(dgm[1], tools="box_select", pt_size=10)
  p.output_backend = "svg"
  # p.rect(**reformat_box([1.48, 1.68, 1.70, 1.88]), fill_color=c1, line_color="black")
  # p.rect(**reformat_box([1.12, 1.28, 1.36, 1.56]), fill_color=c2, line_color="black")
  # p.rect(**reformat_box([1.0, 1.2, 2.0, 2.2]), fill_color=c3, line_color="black")
  # p.rect(**reformat_box([1.44, 1.52, 1.54, 1.68]), fill_color=c4, line_color="black")
  p.x_range = Range1d(0.98, 2.22)
  p.y_range = Range1d(0.98, 2.22)
  # p.title.text = f"Persistence diagram (Elephant w/ {index} triangles)"
  p.title.visible = False
  p.xaxis.visible = False 
  p.yaxis.visible = False 
  p.toolbar_location = None

  # show(p)
  export_svg(p, filename=f"elephants_expsearch/dgm_full.svg")
  export_svg(p, filename=f"elephants_expsearch/dgm_{index}.png")
  # export_png(p, filename=f"elephants_expsearch/dgm_{index}.png")

# %% Elephant filtration animation
mesh_loader = pose_meshes(simplify=2000, which=["elephant"]) # 8k triangles takes ~ 40 seconds
mesh = mesh_loader[0]
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_unreferenced_vertices()
A = geodesics(mesh)
mesh_ecc = A.max(axis=1)
ecc_weight = lower_star_filter(mesh_ecc)
K_ecc = filtration(simplicial_complex(np.asarray(mesh.triangles)), ecc_weight)
dgm = ph(K_ecc, engine="dionysus")

## Show the elpehant with various thresholds of transparency 
mesh.compute_vertex_normals()
# tri_ecc = w(np.array(mesh.triangles))
# tri_colors = bin_color(tri_ecc, "turbo")
normalize_direction(mesh)
mesh.transform(rotation_matrix(0.35*np.pi, [0,1,0]))
vertex_colors = bin_color(w(faces(S,0)))[:,:3]

mesh.vertex_colors = o3.utility.Vector3dVector(vertex_colors)

vis.add_geometry(mesh)
vis.poll_events() ## don't remove this
vis.update_renderer()
# vis.capture_depth_image("elephant_depth.png", False)
vis.capture_screen_image(f"elephants_expsearch/elephant_{i}_{t_size:05d}.png", False)
vis.remove_geometry(mesh)
vis.poll_events() ## don't remove this
vis.update_renderer()


## Use trimesh to color the triangles with transparency
import trimesh
from trimesh.visual.color import ColorVisuals
from trimesh.transformations import rotation_matrix
NV = len(mesh.vertices)
V_XYZ = np.array(mesh.vertices)
V, T = np.arange(NV), np.array(mesh.triangles)
mesh_t = trimesh.Trimesh(vertices=V_XYZ, faces=T)
t_ecc, v_ecc = ecc_weight(mesh_t.faces), np.ravel(ecc_weight(V[:,np.newaxis]))
t_col = (bin_color(t_ecc, "turbo") * 255).astype(np.uint8)
v_col = (bin_color(v_ecc, "turbo") * 255).astype(np.uint8)
mesh_t.show()

import io
import copy
import time
from PIL import Image
# scene = trimesh.scene.Scene()

# sel_vertices = ecc_weight([[v] for v in range(NV)]) <= t
# mesh_t = trimesh.Trimesh(vertices=V[sel_vertices], faces=T[ecc_weight(T) <= t])

## Animation showing the filtration
thresholds = np.linspace(1.01*np.min(t_ecc), np.max(t_ecc), 90)
for i, t in enumerate(thresholds):
  scene = trimesh.scene.Scene()
  mesh_t = trimesh.Trimesh(vertices=V_XYZ, faces=T)
  TC, VC = copy.deepcopy(t_col), copy.deepcopy(v_col)
  TC[t_ecc >= t,-1] = 0
  VC[v_ecc >= t,-1] = 0
  print(f"number of visible vertices: {np.sum(VC[:,-1] != 0)}, triangles: {np.sum(TC[:,-1] != 0)}")
  mesh_t.visual = ColorVisuals(face_colors=TC, vertex_colors=VC)
  # mesh_t.visual.update_faces(ecc_weight(T) <= t) #
  mesh_name = scene.add_geometry(mesh_t, node_name=f"mesh_{i}")
  scene.apply_transform(rotation_matrix(np.pi/2.5, [0,1,0], scene.extents))
  data = scene.save_image(resolution=(400, 400), visible=True)
  time.sleep(2.25)
  image = Image.open(io.BytesIO(data))
  image.save(f'./elephant_animation/frame_{i}.png', format='PNG')
  # scene.delete_geometry(f"mesh_{i}")

# scene.export("./elephant_animation/ani1")

## Learning what the multiplicity does
scene = trimesh.scene.Scene()
mesh_t = trimesh.Trimesh(vertices=V_XYZ, faces=T)
TC, VC = copy.deepcopy(t_col), copy.deepcopy(v_col)
# i,j,k,l = [1.48, 1.68, 1.70, 1.88] ## legs

# ## positive terms 
# VC[:,:] = [255,255,255,255] # reset
# t_ecc <= k
# # VC[v_ecc >= t,:] = [0, 0,255,255]
# TC[t_ecc >= t,-1] = 0

print(f"number of visible vertices: {np.sum(VC[:,-1] != 0)}, triangles: {np.sum(TC[:,-1] != 0)}")
mesh_t.visual = ColorVisuals(face_colors=TC, vertex_colors=VC)
  # mesh_t.visual.update_faces(ecc_weight(T) <= t) #
  mesh_name = scene.add_geometry(mesh_t, node_name=f"mesh_{i}")
  scene.apply_transform(rotation_matrix(np.pi/2.5, [0,1,0], scene.extents))
  data = scene.save_image(resolution=(400, 400), visible=True)
  time.sleep(2.25)
  image = Image.open(io.BytesIO(data))
  image.save(f'./elephant_animation/frame_{i}.png', format='PNG')

# mesh.visual.face_colors = (bin_color(w(faces(S, 2)), "turbo") * 255).astype(np.uint8)

camera_transformation = np.eye(4)
# camera_transformation[:3, :3] = v_prin_dir.T
# scene.apply_transform(camera_transformation)




# o3.visualization.draw_geometries([mesh])

#
# device = o3.core.Device("CPU:0")
# dtype_f = o3.core.float32
# dtype_i = o3.core.int32
# mesh_t = o3.t.geometry.TriangleMesh(device)
# mesh_t.vertex.positions = o3.core.Tensor(np.asarray(mesh.vertices), dtype_f, device)
# mesh_t.triangle.indices = o3.core.Tensor(np.array(mesh.triangles), dtype_i, device)
# mesh_t.vertex.normals = o3.core.Tensor(np.array(mesh.vertex_normals), dtype_f, device)
# mesh_t.vertex['colors'] = o3.core.Tensor(tri_colors, dtype_f, device)








# import open3d.visualization.rendering as rendering
# render = rendering.OffscreenRenderer(640, 480)
# img = render.render_to_image()





# %% Make an animation of the sparsification search
from splex import * 
from pbsig.linalg import pca
from pbsig.color import bin_color
import trimesh
from trimesh.transformations import rotation_matrix

S, w, mesh_ = sparsifier.loader(200)
v_pos = np.asarray(mesh_.vertices)
v_prin_dir = np.linalg.eigh(np.cov(v_pos.T))[1]
mesh = trimesh.Trimesh(v_pos, np.asarray(mesh_.triangles), process=False)

## Color the triangles 
from trimesh.visual.color import ColorVisuals
tri_ecc = w(np.array(mesh_.triangles))
mesh.visual = ColorVisuals(face_colors=(bin_color(tri_ecc, "turbo") * 255).astype(np.uint8))
# mesh.visual.face_colors = (bin_color(w(faces(S, 2)), "turbo") * 255).astype(np.uint8)

camera_transformation = np.eye(4)
camera_transformation[:3, :3] = v_prin_dir.T
scene = trimesh.scene.Scene()
scene.add_geometry(mesh)
scene.apply_transform(camera_transformation)
scene.apply_transform(rotation_matrix(np.pi, [0,0,1], scene.extents))
scene.apply_transform(rotation_matrix(-np.pi/1.6, [0,1,0], scene.extents))

from trimesh.scene.lighting import PointLight, autolight
# scene.lights.append(PointLight(color=[255,255,255], intensity=150.0, radius=150))

scene.lights[0].radius = 10
scene.lights[1].radius = 10
scene.lights[0].color = np.array([255,255,255,255], dtype=np.uint8)
scene.lights[1].color = np.array([255,255,255,255], dtype=np.uint8)
scene.lights[0].intensity = 1e15
scene.lights[1].intensity = 1e15
scene.show()

# %% Try PyRender instead since trimesh is limited and open3d doesnt support headless screenshots
import pyrender 
mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth = False)

scene = pyrender.Scene(
    ambient_light=[0.02, 0.02, 0.02], 
    bg_color=[1.0, 1.0, 1.0]
)
# light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)

scene.add_node(pyrender.Node(mesh=mesh_pr, matrix=np.eye(4)))
scene.add_node(pyrender.Node(cam, matrix=np.eye(4)))

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
       [0.0, -s,   s,   0.3],
       [1.0,  0.0, 0.0, 0.0],
       [0.0,  s,   s,   0.35],
       [0.0,  0.0, 0.0, 1.0],
    ])
scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
r.render(scene)

# INteractive 
# from pyrender.viewer import Viewer
# Viewer(scene)


# import os
# os.environ['PYOPENGL_PLATFORM'] = 'Pyglet'
# r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
# color, depth = r.render(scene)



## Save image (headless doesnt quite work but that's ok)
import io
from PIL import Image
data = scene.save_image(resolution=(400,400), visible=False)
image = Image.open(io.BytesIO(data))


# p = figure(width=400, height=400)
# p.x_range.range_padding = p.y_range.range_padding = 0
# p.scatter(np.arange(10), np.arange(10), color="red")
# # p.image_rgba(image=[image], x=0, y=0, dw=10, dh=10)
# p.grid.grid_line_width = 0.5
# show(p)

# %% Expo search testing 
# I = []
# z = np.zeros(80000, dtype=bool)
# z[912:] = True
# first_true_exp(z, k=1, intervals=I)
# I





# %%  Data set 
from pbsig.datasets import noisy_circle
X = noisy_circle(50, n_noise=10, perturb=0.10, r=0.10)
p = figure(width=250, height=250)
p.scatter(*X.T)
show(p)

# %% Use the insertion radius of the greedy permuttion 
from pbsig.shape import landmarks
from pbsig.persistence import ph
pi, insert_rad = landmarks(X, k=len(X), diameter=True)

# %% Construct the 2-d rips compelx at the enclosing radius to true diagram
from pbsig.vis import figure_dgm
from splex.geometry import rips_complex, enclosing_radius
er = enclosing_radius(X)
S = rips_complex(X, radius=er, p=2)
K = rips_filtration(X, radius=er, p=2)
dgm = ph(K, engine="dionysus")

# %% Show the diagram 
p = figure_dgm(dgm[1])
show(p)

# %% Test what greedypermutation's prefix points look like
dgms_sparse = [ph(rips_filtration(X[pi[:i]], radius=er, p=2), engine="dionysus") for i in range(10, 60, 5)]
dgm_figs = []
for dgm in dgms_sparse:
  p = figure_dgm(dgm[1], width=200, height=200)
  p.x_range = Range1d(0, 1.1)
  p.y_range = Range1d(0, 1.1)
  dgm_figs.append(p)

show(row(*dgm_figs))

# %% Simple exponential search 
x = np.sort(np.random.choice(range(1500), size=340, replace=False))
v = 78

## Needs: 
##  1. Sequence with __len__ + __getitem__() support
##  2. target type with __eq__ and __le__ support
def binary_search(seq: Sequence, target: Any):
  left, right = 0, len(seq) - 1
  while left <= right:
    mid = left + (right - left) // 2
    val = seq[mid]
    if val == target:
      return mid
    left, right = (mid + 1, right) if val < target else (left, mid - 1)
  return None # Element not found
 
assert np.all([binary_search(x, x[i]) == i for i in range(len(x))])

## TODO: augment to work only for '<=' or better for arbitrary predicate
## Also augment to work with binary search as well  
# goal is to find the index 'i' right before the predicate returns False, i.e.
# TRUE, TRUE, TRUE, ..., TRUE, TRUE,  FALSE, FALSE, ..., FALSE
# ------------------------------(i)^--(i+1)^------------------ 
def exponential_search(seq: Sequence, target: Any, l: int = 0):
  print(f"({l})")
  if seq[l] == v:
    return l
  n = 1
  while (l+n) < len(seq) and seq[l+n] <= v:
    #print(l+n)
    n *= 2
  l = min(l + (n // 2), len(seq) - 1) # min(l+n, len(arr) - 1)
  return exponential_search(seq, v, l)

assert np.all([exponential_search(x, x[i]) == i for i in range(len(x))])

exponential_search(np.array([True, True, True, True, False, False]), False)

## Also augment to work with binary search as well  
# goal is to find the index 'i' right before the predicate returns False, i.e.
# TRUE, TRUE, TRUE, ..., TRUE, TRUE,  FALSE, FALSE, ..., FALSE
# ------------------------------(i)^--(i+1)^------------------ 
def first_true(seq: Sequence):
  left, right = 0, len(seq) - 1
  val = False
  while left < right:
    mid = left + (right - left) // 2
    val = seq[mid]
    print(mid)
    if val is False: 
      left = mid + 1
    else:
      if mid == 0 or seq[mid - 1] is False:
        return mid
      right = mid - 1
  return left if val else -1

L = [False]*1345 + [True]*1500
first_true_exp(L)

from array import array
class CostSearch():
  def __init__(self, L) -> None:
    self.L = L
    self.accesses = 0
    self.indices = array('I')
    self.resets = array('I')

  def __getitem__(self, i: int):
    self.accesses += 1
    self.indices.append(i)
    return self.L[i]
  
  def resolve(self):
    self.resets.append(self.accesses)
  
  def __len__(self):
    return len(self.L)
  
LC = CostSearch(L)
first_true_exp(LC, k=2)
LC.accesses


# %% Skew access and index cost testing
from scipy.stats import skewnorm
z = np.sort(np.floor(skewnorm(10, loc=2, scale=10).rvs(1500))).astype(np.int32)
print(str(np.histogram(z)[0]))

m: float = 0.2
index_costs, access_costs = [], []
for i in np.linspace(0, max(z)/3, 30):
  Z = CostSearch(z >= i)
  index = first_true_exp(Z, k=10) # change k for binary vs exponential search 
  assert index == np.flatnonzero(z >= i)[0]
  index_costs.append(np.sum(m*np.array(Z.indices)))
  access_costs.append(Z.accesses)

## For heavily skewed samples the recursive exp search is indeed much faster in terms of index costs
## though the skewness needs to be quite extreme
print(f"Mean index cost: {np.sum(index_costs)/30:.2f}, Mean access cost: {np.sum(access_costs)/30:.2f}")

# if (val and lb == 0) or (val and lb == ub): 
#   return lb 
# elif val == True and lb != ub: 
#   return lb 
# else: 
#   lb = lb * 2
#   first_true_exp


# def first_true_exp(seq: Sequence, ub: int = 1):
#   print(ub)
#   if ub >= len(seq): ## we know its in the last half then
#     lb = ub // 2

#   if seq[ub] is True:
#     if ub == 0 or seq[ub - 1] is False:
#       return ub
#     else:
#       return first_true_exp(seq, ub * 2)
#   else:
#     return first_true_exp(seq, ub * 2 + 1)
    
L = [False]*1345 + [True]*15000

first_true(L)


# %% Problem: using landmark points for subsampled rips is still a rips complex, which is intractable....
# Need an alternative complex or data set... maybe delaunay...
# Or just use the given sparse mesh. Take an elephant, choose landmarks/greedypermutation, then...
# 1) Consider problem of *generating* a sparse filtered complex from a *point set alone*---start with the true diagram, and 
# then use exp search on landmark points until a given multiplicity is satisfied (say, w/ geodesic eccentricity filter + delaunay complex)
# UPDATE: this seems not feasible as delaunay ecc wont generate the right dgm as the structured mesh
# 2) Reverse problem: You have a mesh + it's true diagram (from mesh!), you wish to sparsify (say w/ quadric simplification) 
# as much as possible, but under the constraint it preserves a given topology. Again, eccentricity + elephant constraint. 
# 
# could also *perhaps* formulate a rips/based or spectra based objective for sparsification

# %% (1) Load mesh and do floyd warshall to get geodesic distances
from pbsig.datasets import pose_meshes
# from pbsig.shape import landmarks
# mesh_loader = pose_meshes(simplify=20000, which=["elephant"])
# elephant_mesh = mesh_loader[0]
# elephant_mesh.euler_poincare_characteristic() # 2 
# elephant_mesh.is_watertight() # False 

# elephant_pos = np.asarray(elephant_mesh.vertices)
# lm_ind, lm_radii = landmarks(elephant_pos, k=len(elephant_pos))




# ecc_weight = lower_star_filter(mesh_ecc)
# S = simplicial_complex(np.asarray(elephant_mesh.triangles))
# sieve = Sieve(S, family=[ecc_weight], p = 1)
# sieve.pattern = np.array([[1.48, 1.68, 1.70, 1.88], [1.12, 1.28, 1.36, 1.56], [1.0, 1.2, 2.0, 2.2]])
# sieve.sift()
# box_counts = np.ravel(sieve.summarize(spectral_rank))
# np.allclose(box_counts, [4,2,1])






# %% Run exponential search 


# %% (2) Does delaunay recover this?
from pbsig.persistence import ph
from pbsig.vis import figure_dgm
dgm = ph(K_ecc, engine="dionysus")

p = figure_dgm(dgm[1])
show(p)

from pbsig.linalg import up_laplacian
from scipy.sparse import diags
S = delaunay_complex(np.asarray(elephant_mesh.vertices))

L = up_laplacian(S, p=0)
A = L - diags(L.diagonal())
A = A.tocoo()
A.data = np.linalg.norm(elephant_pos[A.row] - elephant_pos[A.col], axis=1)
#A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
AG = floyd_warshall(A.tocsr())
mesh_ecc = AG.max(axis=1)
ecc_weight = lower_star_filter(mesh_ecc)


K_ecc = filtration(S, ecc_weight)
dgm = ph(K_ecc, engine="dionysus")
p = figure_dgm(dgm[1])
show(p)





# %% Precompute eccentricities
mesh_ecc = [squareform(gd).max(axis=1) for gd in mesh_geodesics]
ecc_f = [[max(mesh_ecc[cc][i], mesh_ecc[cc][j]) for (i,j) in combinations(range(X.shape[0]), 2)] for cc, (X, mesh) in enumerate(meshes)]
eff_f = [np.array(ecc) for ecc in ecc_f]


# %% Use gudhi's sparse rips 



# %% Compute the weights for each vertex
# https://arxiv.org/pdf/1306.0039.pdf
# https://www.cs.purdue.edu/homes/tamaldey/book/CTDAbook/CTDAbook.pdf
# https://research.cs.queensu.ca/cccg2015/CCCG15-papers/01.pdf 
from scipy.spatial.distance import pdist
# eps = 0.1 # 0 < eps < 1
# def sparse_weight(X: np.ndarray, radii: np.ndarray, eps: float) -> Callable:
#   pd = pdist(X)
#   def _vertex_weight(i: int, alpha: float):
#     if (radii[i]/eps) >= alpha: return 0.0
#     if (radii[i]/(eps-eps**2)) <= alpha: return eps*alpha
#     return alpha - radii[i]/eps
  
#   def _weight(s: SimplexConvertible, alpha: float):
#     if dim(s) == 0: 
#       return _vertex_weight(s[0], alpha) 
#     elif dim(s) == 1:
#       return pd[rank_lex(s, n=len(X))] + _vertex_weight(s[0], alpha) + _vertex_weight(s[1], alpha)
#     else:
#       return np.max([_weight(f, alpha) for f in faces(s, p=1)])
#   return _weight

def sparse_rips_filtration(X: ArrayLike, L: np.ndarray, epsilon: float, p: int = 1, max_radius: float = np.inf, remove_singletons: bool = True):
  """Sparse rips complex """
  assert epsilon > 0, "epsilon parameter must be positive"

  d = pdist(X)
  # np.digitize(x, bins=[0.33, 0.66, 1])

  ## Step 0: Make a vectorized edge birth time function 
  def edge_birth_time(edges: ArrayLike, eps: float, n: int) -> np.ndarray:
    ## Algorithm 3 from: https://arxiv.org/pdf/1506.03797.pdf
    edge_ind = np.array(rank_combs(edges, order='lex', n=n))
    L_min = L[edges].min(axis=1)
    birth_time = np.ones(len(edges))*np.inf # infinity <=> don't include in complex
    cond0 = d[edge_ind] <= 2 * L_min * (1+eps)/eps
    cond1 = d[edge_ind] <= L[edges].sum(axis=1) * (1+eps)/eps
    birth_time[cond1] = d[edge_ind[cond1]] - L_min[cond1]*(1+eps)/eps
    birth_time[cond0] = d[edge_ind[cond0]] / 2
    return birth_time

  ## Step 1: Construct the sparse graph of the input point cloud or metric X 
  n = len(X)
  all_edges = np.array(list(combinations(range(n), 2))) ## todo: add option to use kd tree or something
  edge_fv = edge_birth_time(all_edges, epsilon, n)
  S_sparse = simplicial_complex(chain([[i] for i in range(n)], all_edges[edge_fv != np.inf]))

  ## Step 2: Perform a sparse k-expansion, if 2-simplices or higher needed 
  if p > 1: 
    pass 


  # eps = ((1.0-epsilon)/2) * epsilon # gudhi's formulation
  # edges = np.array(faces(G, 1))
  # edge_weights = (L[edges]).max(axis=1)
  # edge_sparse_ind = np.flatnonzero(edge_weights >= alpha)
  # edge_weights = (L[edges[edge_sparse_ind]]).min(axis=1)*eps
  # return edges[edge_sparse_ind], edge_weights

# ([0, 1], 0.14069064627395436) == pdist(X[[0,1],:])

# %% Test sparsified complexes 
from gudhi import RipsComplex
S_rips = RipsComplex(points=X, max_edge_length=er/2, sparse=0.90)
st = S_rips.create_simplex_tree()
S_sparse = simplicial_complex([Simplex(s[0]) for s in st.get_simplices()])
#  If `block_simplex` returns true, the simplex is removed, otherwise it is kept. 

ES_edges, ES_weight = sparse_weight(S, insert_rad, 0.90, 2*er)

card(S,1)
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.50, er)[0])
len(sparse_weight(S, insert_rad, 0.90, 2*er)[0])
S_sparse = simplicial_complex()
S_sparse.update([[i] for i in range(card(S,0))])
S_sparse.update(sparse_weight(S, insert_rad, 0.90, er)[0])






# %% Show the sparsified complex
from pbsig.vis import figure_complex
p = figure_complex(S_sparse, pos=X)
show(p)

# %% Run exponential search 
R = (0, 0.3, 0.6, 1.0)


# f_sparse = sparse_weight(X, insert_rad, 0.1)

# face_weights = np.array([f_sparse(s,) for s in faces(S)])


# %% Run exponential search 


# %%  Fix alpha, optimize epsilon
# TODO: insert points one at a time by non-zero additions to matvec operator...
from functools import partial
ir = np.array(insert_rad)[np.argsort(pi)] # 
wf = sparse_weight(X, ir, eps=1-1e-12)
wf = sparse_weight(X, ir, eps=0.00001)
K = filtration(S, partial(wf, alpha=er))

sum(np.array(list(K.indices())) <= 2*er)


from pbsig.vis import figure_dgm
dgm = ph(K, engine="dionysus")
p = figure_dgm(dgm[1])
show(p)


# %% 

# %% Binary / exp search skecthing
costs = []
z = np.zeros(1500, dtype=bool)
for i in range(len(z)-1):
  # z[i] = True 
  z[-(i+1)] = True 
  Z = CostSearch(z)
  first_true_exp(Z, k=9)
  costs.append(Z.cost)
print(np.sum(costs))

costs = []
for i in range(1500):
  z = np.zeros(1500, dtype=bool)
  j = np.random.choice(1500, size=1, p=)

# Doesnt change after at k = np.log2(n) splits

## Check is correct 
z = np.zeros(1500, dtype=bool)
first_true_exp(z, 0, len(z)-1)
for i in range(len(z)-1):
  z[-(i+1)] = True 
  assert first_true_exp(z, 0, len(z)-1) == (len(z) - i - 1)


z = np.zeros(1500, dtype=bool)
assert first_true_exp(z) == -1
for i in range(len(z)-1):
  z[-(i+1)] = True 
  assert first_true_exp(z) == (len(z) - i - 1)

print(first_true_exp(z, lb=0, ub=len(L)-1))

z = np.zeros(1500, dtype=bool)
first_true_bin(z, 0, len(z)-1)
for i in range(len(z)-1):
  z[-(i+1)] = True 
  assert first_true_bin(z, 0, len(z)-1) == (len(z) - i - 1)

first_true_bin(L, 0, len(L)-1)

def binary_search_first(seq: Sequence, lb: int = 0, ub: int = None):
  lb, ub = max(0, lb), len(seq) - 1 if ub is None else min(ub, len(seq) - 1)
  while lb <= ub:
    mid = lb + (ub - lb) // 2
    val = seq[mid]
    if val:
      return mid
    left, right = (mid + 1, right) if val < target else (left, mid - 1)
  return None # Element not found












# %% 
# from metricspaces import MetricSpace
# from greedypermutation import clarksongreedy, Point
# point_set = [Point(x) for x in X]
# M = MetricSpace(point_set)
# G = list(clarksongreedy.greedy(M))
# G_hashes = [hash(g) for g in G]
# P_hashes = [hash(p) for p in point_set]


