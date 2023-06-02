# %% Imports 
from itertools import *
import networkx as nx
import numpy as np
import open3d as o3
from more_itertools import chunked, pairwise
from pbsig.betti import Sieve
from pbsig.linalg import eigvalsh_solver
from scipy.sparse.csgraph import floyd_warshall
from splex import *

# %% Bokeh configuration 
import bokeh
from bokeh.plotting import show, figure 
from bokeh.models import Range1d
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()

# %% Configure the mesh directories 
import os 
mesh_dir = "/Users/mpiekenbrock/pbsig/src/pbsig/data/mesh_poses"
pose_dirs = ["camel-poses", "elephant-poses", "horse-poses", "flamingo-poses"]
get_pose_objs = lambda pd: np.array(list(filter(lambda s: s[-3:] == "obj", sorted(os.listdir(mesh_dir+"/"+pd)))))
pose_objs = [get_pose_objs(pose_type)[[0,1,2,3]] for pose_type in pose_dirs]
pose_objs = [[obj_type + "/" + o for o in objs] for obj_type, objs in zip(pose_dirs, pose_objs)]
pose_objs = list(collapse(pose_objs))
pose_paths = [mesh_dir + "/" + obj for obj in pose_objs]

# %% TODO: try SHREC20 
# "/Users/mpiekenbrock/neuromorph/data/meshes/SHREC20b_lores/full"


# %% Lazily-load the mesh data, normalized
import networkx as nx
from pbsig.itertools import LazyIterable
from pbsig.pht import normalize_shape, stratify_sphere
def load_mesh(i: int):
  mesh_in = o3.io.read_triangle_mesh(pose_paths[i])
  mesh_in.compute_vertex_normals()
  mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=5000)
  X = np.asarray(mesh_smp.vertices)
  V = stratify_sphere(2, 64)
  X_norm = normalize_shape(X, V=V, translate="hull")
  return X_norm, mesh_smp
meshes = list(LazyIterable(load_mesh, len(pose_paths)))

# %% Profile 
# from pbsig.pht import archimedean_sphere, shape_center
# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(load_mesh)
# profile.add_function(stratify_sphere)
# profile.add_function(normalize_shape)
# profile.add_function(shape_center)
# profile.add_function(archimedean_sphere)
# profile.enable_by_count()
# meshes = list(LazyIterable(load_mesh, 1))
# profile.print_stats(output_unit=1e-3, stripzeros=True)

# %% Compute geodesic distances
def geodesic_dist(X, mesh):
  mesh.compute_adjacency_list()
  G = nx.from_dict_of_lists({ i : list(adj) for i, adj in enumerate(mesh.adjacency_list) })
  A = nx.adjacency_matrix(G)
  A.data = np.array([np.linalg.norm(X[i,:] - X[j,:]) for i,j in zip(*A.nonzero())], dtype=np.float32)
  AG = floyd_warshall(A)
  return squareform(AG)
mesh_geodesics = [geodesic_dist(X, mesh) for X, mesh in meshes]

# %% Precompute eccentricities
mesh_ecc = [squareform(gd).max(axis=1) for gd in mesh_geodesics]
ecc_f = [[max(mesh_ecc[cc][i], mesh_ecc[cc][j]) for (i,j) in combinations(range(X.shape[0]), 2)] for cc, (X, mesh) in enumerate(meshes)]
eff_f = [np.array(ecc) for ecc in ecc_f]

# %% Compute eccentricity-equipped diagrams 
from pbsig.persistence import ph
from splex.geometry import flag_weight
# ecc_p = [0,0.01] + list(np.linspace(0.1, 1, 9)) + list(range(1,6)) + [10,30,50]
dgms = []
for i, (X, mesh) in enumerate(meshes):
  S = simplicial_complex(mesh.triangles)
  diam_filter = flag_weight(np.maximum(mesh_geodesics[i], 0.5*eff_f[i]))
  K = filtration(S, f=diam_filter)
  dgms.append(ph(K, engine="dionysus"))

# %% Visualize the bottleneck-based distance matrix for the enriched diagrams
import gudhi
from pbsig.vis import figure_dist, figure_dgm
p_hom = 1
to_mat = lambda d: np.c_[d['birth'][d['death'] != np.inf], d['death'][d['death'] != np.inf]]
bd = np.array([gudhi.bottleneck_distance(to_mat(dgms[i][p_hom]), to_mat(dgms[j][p_hom])) for i,j in combinations(range(len(dgms)), 2)])
dgm_figs = [figure_dgm(d[1], width=150, height=150) for d in dgms]
show(row(*dgm_figs))
show(figure_dist(bd))

# %% Randomly maxmin sample to get sieve pattern
dgm_colors = bokeh.palettes.viridis(len(dgms))
p = figure()
for d, dcol in zip(dgms, dgm_colors):
  birth, death = d[1]['birth'], d[1]['death']
  dgm_points = np.c_[birth[death != np.inf], death[death != np.inf]]
  p.scatter(*dgm_points.T, color=dcol)
show(p)

b = np.hstack([d[1]['birth'] for d in dgms])
d = np.hstack([d[1]['death'] for d in dgms])
dgm_points = np.c_[b[d != np.inf], d[d != np.inf]]
D = squareform(pdist(dgm_points))
knn = np.argsort(D, axis=1)[:,5]
L0 = np.argmin([D[i,j] for i,j in enumerate(knn)])
L = np.array([L0])
for i in range(16):
  dl = D[:,L]
  dl[L,:] = -np.inf
  L = np.append(L, [np.argmax(D[:,L].min(axis=1))])

p = figure()
p.scatter(*dgm_points.T)
p.scatter(*dgm_points[L,:].T, color="red")
show(p)


from pbsig.betti import sample_rect_halfplane
np.random.seed(1234)
P = dgm_points[L,:]
area = np.prod(np.abs(dgm_points.min(axis=0) - dgm_points.max(axis=0)))
rects = sample_rect_halfplane(len(P), lb=area*0.025, ub=area*0.10)
a = P[:,0] - (abs(rects[:,0] - rects[:,1]))/2
b = P[:,0] + (abs(rects[:,0] - rects[:,1]))/2
c = P[:,1] - (abs(rects[:,2] - rects[:,3]))/2
d = P[:,1] + (abs(rects[:,2] - rects[:,3]))/2
rects = np.array(list(zip(a,b,c,d)))
rects = rects[np.logical_and(rects[:,0] <= rects[:,1], rects[:,1] <= rects[:,2], rects[:,2] <= rects[:,3]),:]


# %% Select a random sieve pattern and 
sieves = []
for i, (X, mesh) in enumerate(meshes):
  X, mesh = meshes[i]
  S = simplicial_complex(mesh.triangles, form="tree")
  # diam_filters = [flag_weight(np.maximum(mesh_geodesics[i], r*eff_f[i])) for r in [0.0, 0.01, 0.1, 0.2, 0.5, 5.0]]
  f = flag_weight(np.maximum(mesh_geodesics[i], 0.5*eff_f[i]))
  sieve = Sieve(S, [f], p=1)
  sieve.pattern = rects
  # sieve.randomize_pattern(4) # show(sieve.figure_pattern())
  sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
  sieves.append(sieve)

for sieve in sieves: 
  sieve.sift(w=0.35, k=25)

p = sieves[0].figure_pattern()
p.x_range = Range1d(min(sieves[0].pattern['i'])-1, max(sieves[0].pattern['j'])+1)
p.y_range = Range1d(min(sieves[0].pattern['i'])-1, max(sieves[0].pattern['j'])+1)
show(p)

summaries = [sieve.summarize() for sieve in sieves]
nd = np.array([np.linalg.norm(summaries[i] - summaries[j]) for i,j in combinations(range(len(summaries)), 2)])
show(figure_dist(nd))

import time
from pbsig.linalg import eigen_dist
from math import comb
normalize = lambda x: (x - min(x))/(max(x) - min(x))
nd = np.zeros(comb(len(sieves), 2))
# nd.fill(np.inf)
for cc in range(40):
  d_ij = np.array([eigen_dist(sieves[i].spectra[cc]['eigenvalues'], sieves[j].spectra[cc]['eigenvalues']) for i,j in combinations(range(len(sieves)), 2)])
  nd = d_ij
  show(figure_dist(nd))
  time.sleep(0.5)
  
  # nd += d_ij
  # nd += np.where(normalize(d_ij) <= 0.25, normalize(d_ij), 1.0)
  # nd += normalize(d_ij)
  # nd = np.minimum(nd, d_ij)
show(figure_dist(nd))

# %% Try to optimize a convex combination of corner points 
normalize = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
n_corner_pts = len(sieves[0].spectra)
spectral_dist = lambda cc: np.array([eigen_dist(sieves[i].spectra[cc]['eigenvalues'], sieves[j].spectra[cc]['eigenvalues']) for i,j in combinations(range(len(sieves)), 2)])
D = [spectral_dist(cc) for cc in range(n_corner_pts)]

def convex_distance(alpha: np.ndarray):
  fd = np.zeros(comb(len(sieves), 2))
  for i, a in enumerate(alpha): 
    fd += a * D[i]
  return fd

def bottleneck_loss(alpha: np.ndarray):
  fd = convex_distance(alpha)
  return np.sum((normalize(fd) - normalize(bd))**2)

from scipy.optimize import minimize, LinearConstraint
sum_to_one = LinearConstraint(np.ones(40)[np.newaxis,:], lb=1, ub=1)
res = minimize(bottleneck_loss, x0=np.ones(40)/40, constraints=sum_to_one, method="trust-constr", bounds=[(0,1)])
nd = convex_distance(res.x)
show(figure_dist(nd))







# %% 
## Compare the distances 
from bokeh.io import output_notebook
from bokeh.plotting import show
from pbsig.dsp import signal_dist
from pbsig.vis import figure_dist
from scipy.spatial.distance import pdist, squareform

output_notebook()
summaries = [sieve.summarize() for sieve in sieves]
D = np.array([signal_dist(summaries[i][0], summaries[j][0], scale=True, center=True) for i,j in combinations(range(len(sieves)), 2)])
show(figure_dist(D))

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


# %%

## Plot the low-d embeddings 
# from pbsig.linalg import cmds
# p = figure()
# p.scatter(*cmds(squareform(mesh_geodesics[5]**2), d=2).T)
# show(p)

