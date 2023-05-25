import trimesh 
import numpy as np
import networkx as nx
import open3d as o3

from splex import *
from scipy.sparse.csgraph import floyd_warshall
from itertools import * 
from more_itertools import pairwise, chunked
from pbsig.linalg import eigvalsh_solver
from pbsig.betti import Sieve
# mesh = trimesh.load("/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses/elephant-poses/elephant-01.obj")

mesh_dir = "/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses"
camel_poses = [f"camel-0{i}.obj" for i in range(1,4)] + ["camel-reference.obj"]
camel_poses = ["camel-poses/"+fp for fp in camel_poses]
elephant_poses = [f"elephant-0{i}.obj" for i in range(1,4)] + ["elephant-reference.obj"]
elephant_poses = ["elephant-poses/"+fp for fp in elephant_poses]
pose_paths = [mesh_dir + "/" + path for path in chain(elephant_poses, camel_poses)]

## Preprocess 
from pbsig.pht import stratify_sphere, normalize_shape
from pbsig.itertools import LazyIterable

def load_mesh(i: int):
  mesh_in = o3.io.read_triangle_mesh(pose_paths[i])
  mesh_in.compute_vertex_normals()
  mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=5000)
  X = np.asarray(mesh_smp.vertices)
  V = stratify_sphere(2, 132)
  X_norm = normalize_shape(X, V)
  return X_norm, mesh_smp

import networkx as nx
meshes = list(LazyIterable(load_mesh, len(pose_paths)))

def geodesic_dist(X, mesh):
  mesh.compute_adjacency_list()
  G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(mesh.adjacency_list) })
  A = nx.adjacency_matrix(G)
  A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
  AG = floyd_warshall(A)
  return squareform(AG)

mesh_geodesics = [geodesic_dist(X, mesh) for X, mesh in meshes]
mesh_ecc = [squareform(gd).max(axis=1) for gd in mesh_geodesics]


## Plot the low-d embeddings 
# from pbsig.linalg import cmds
# p = figure()
# p.scatter(*cmds(squareform(mesh_geodesics[5]**2), d=2).T)
# show(p)


from splex.geometry import flag_weight
from pbsig.persistence import ph 


R = [0,0.01] + list(np.linspace(0.1, 1, 9)) + list(range(1,6)) + [10,30,50]
ecc_f = [[max(mesh_ecc[cc][i], mesh_ecc[cc][j]) for (i,j) in combinations(range(X.shape[0]), 2)] for cc, (X, mesh) in enumerate(meshes)]
eff_f = [np.array(ecc) for ecc in ecc_f]

dgms = []
for i, (X, mesh) in enumerate(meshes):
  S = simplicial_complex(mesh.triangles)
  diam_filter = flag_weight(np.maximum(mesh_geodesics[i], 5.0*eff_f[i]))
  K = filtration(S, f=diam_filter)
  dgms.append(ph(K, engine="dionysus"))

import gudhi 
p_hom = 1
to_mat = lambda d: np.c_[d['birth'][d['death'] != np.inf], d['death'][d['death'] != np.inf]]
bd = np.array([gudhi.bottleneck_distance(to_mat(dgms[i][p_hom]), to_mat(dgms[j][p_hom])) for i,j in combinations(range(len(dgms)), 2)])

from pbsig.vis import figure_dist
show(figure_dist(bd))

from pbsig.vis import figure_dgm
p = figure_dgm(dgms[5][1])
show(p)



##
i = 0
X, mesh = meshes[i]
S = simplicial_complex(mesh.triangles)
diam_filters = [flag_weight(np.maximum(mesh_geodesics[i], r*eff_f[i])) for r in [0.0, 0.01, 0.1, 0.2, 0.5, 5.0]]
sieve = Sieve(S, diam_filters)
sieve.randomize_pattern(4) # show(sieve.figure_pattern())
sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
sieve.sift(w=0.15, k=15)



sieves = []
for pose_path in pose_paths: 
  mesh_in = o3.io.read_triangle_mesh("/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses/camel-poses/camel-02.obj")
  mesh_in.compute_vertex_normals()
  mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=5000)
  # o3.visualization.draw_geometries([mesh_smp])

  ## Compute the sieve 
  from pbsig.pht import stratify_sphere, normalize_shape
  X = np.asarray(mesh_smp.vertices)
  V = stratify_sphere(2, 132)
  X_norm = normalize_shape(X, V)

  ## Fix the pattern
  sieve_pattern = np.loadtxt("/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses/elephant-poses/4_pattern.txt")
  sieve_pattern = np.array([tuple(s) for s in sieve_pattern], dtype=[('i', float), ('j', float), ('sign', int), ('index', int)])
  
  ## Construct the sieve
  S = simplicial_complex(np.array(mesh_smp.triangles))
  DV_family = [lower_star_weight(X_norm @ v) for v in V]
  sieve = Sieve(S, DV_family)
  sieve._pattern = sieve_pattern
  sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
  sieve.sift(w=0.15, k=15)

  ## Save the sieve
  sieves.append(sieve)

## Compare the distances 
from bokeh.plotting import show
from bokeh.io import output_notebook
from scipy.spatial.distance import pdist, squareform
from pbsig.dsp import signal_dist
from pbsig.vis import figure_dist
output_notebook()
summaries = [sieve.summarize() for sieve in sieves]
D = np.array([signal_dist(summaries[i][0], summaries[j][0], scale=True, center=True) for i,j in combinations(range(len(sieves)), 2)])
show(figure_dist(D))

def rho_dist(x: np.ndarray, y: np.ndarray, p: int = 2):
  n = max(len(x), len(y))
  a,b = np.zeros(n), np.zeros(n)
  a[:len(x)] = np.sort(x)
  b[:len(y)] = np.sort(y)
  denom = (np.abs(a)**p + np.abs(b)**p)**(1/p)
  return np.sum(np.where(np.isclose(denom, 0, atol=1e-15), 0, np.abs(a-b)/denom))

shape0_p0 = list(chunked(sieves[0].spectra[0]['eigenvalues'], 15))
shape1_p0 = list(chunked(sieves[7].spectra[0]['eigenvalues'], 15))

np.sum([rho_dist(ew1, ew2) for ew1, ew2 in zip(shape0_p0, shape1_p0)])

ev1 = [s['eigenvalues'] for s in chunked(sieves[0].spectra.values(), 15)]
ev2 = [s['eigenvalues'] for s in chunked(sieves[1].spectra.values(), 15)]

# sieve.randomize_pattern(4)
# show(sieve.figure_pattern())
# np.savetxt("/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses/elephant-poses/4_pattern.txt", sieve._pattern)

## Show the summary statistics
summary = sieve.summarize()
n_summaries = len(np.unique(sieve._pattern['index']))
n_dir = len(V)

import time
figs = []
for j in range(n_summaries):
  x = np.arange(n_dir)
  y = summary[j]
  sig = figure_signal(x, y, figkwargs=dict(width=250, height=80), line_width=2)
  sig.toolbar_location = None
  # sig.y_range = Range1d(*bnds)
  sig.xaxis.visible = False
  sig.yaxis.minor_tick_line_color = None  
  figs.append(sig)
  # time.sleep(0.15)s

from bokeh.models import Div
from bokeh.layouts import row, column
show(column(Div(text="Camel-2"), *figs))

## make function-endowed Rips 










# A = nx.adjacency_matrix(mesh.vertex_adjacency_graph)
# X = np.array(mesh.vertices)
# A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
# floyd_warshall(A)
# maxmin sampling


from pbsig.betti import Sieve

S = simplicial_complex(mesh.faces)