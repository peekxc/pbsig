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