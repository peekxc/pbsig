import numpy as np
import trimesh
import bokeh
from bokeh.plotting import figure, show 
from bokeh.layouts import row, column
from bokeh.io import output_notebook
from bokeh.models import Range1d
# output_notebook()

from splex import *
from pbsig.betti import Sieve
from pbsig.pht import parameterize_dt
from pbsig.linalg import *
from os import listdir

class ShapeFamily:
  def __init__(self, obj_files, nd: int = 50):
    self.files = obj_files # shark_dir+obj
    self.nd = nd # number of directions

  def load_mesh(self, index: int = 0):
    mesh = trimesh.load(self.files[index], force='mesh')
    F = np.array(mesh.faces)
    F = np.array([np.sort(p) for p in F])
    F = F[np.lexsort(np.rot90(F)),:]
    return simplicial_complex(mesh.faces)

  def __iter__(self) -> Iterator[Callable]:
    for obj in self.files:
      mesh = trimesh.load(obj, force='mesh')
      DT = parameterize_dt(np.array(mesh.vertices), self.nd, nonnegative=False)
      yield from DT

  def __len__(self) -> int:
    return self.nd * len(self.files)

## Sharks provide the models, the DT provides the direction vectors - the product provides the family
shark_dir = "/Users/mpiekenbrock/Downloads/Shark002.blend/frames/"
shark_frames = sorted(list(filter(lambda s: s[0] == 's' and s[-3:] == "obj", listdir(shark_dir))))
family = ShapeFamily([shark_dir + sf for sf in shark_frames], 50)

## Create the sieve with a random pattern 
S = family.load_mesh(0)
sieve = Sieve(S, family)
# sieve.randomize_pattern(2)

## Choose a sensible tolerance and sift through the family
sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
sieve.sift(progress=True, k=20) # most expensive line

# shark_results = {}
# for obj in shark_frames:
#   mesh = trimesh.load(shark_dir+obj, force='mesh')
global_pattern = sieve._pattern
shark_results = {}
  #mesh.show()



import os
import pyrender
# os.environ['PYOPENGL_PLATFORM'] = 'pyglet'

mesh = pyrender.Mesh.from_trimesh(trimesh.load(family.files[0], force='mesh'))
scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
scene.add(mesh, pose=np.eye(4))
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
scene.add(cam)
pyrender.Viewer(scene)
#color, depth = r.render(scene)

r = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480,point_size=1.0)



import bokeh 
from bokeh.plotting import figure, show 
img = color.view(dtype=np.uint8).reshape((480,640,3)) 
p = figure()
m = prod(img.shape)
p.image_rgba(image=np.ravel(img), x=[0]*m, y=[0]*m, dw=[10]*m, dh=[10]*m)
show(p)
# light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
# 
# 

  # sieve.randomize_pattern(4)
# sieve._pattern = global_pattern



  ## Collect the results 
# shark_results[obj] = sieve

import pickle 
shark_data = { k : v.spectra for k,v in shark_results.items() }
with open(f'shark_signatures.pkl', 'wb') as f:
  pickle.dump(shark_data, f)
# for k,v in shark_results.items():
#   with open(f'shark_signatures.pkl', 'wb') as f:
#     pickle.dump(v.spectra, f)

## Show the pattern 
show(sieve.figure_pattern())

## Show the signatures 
sieve = shark_results['sharkfixed12.obj']
summary = sieve.summarize()
from bokeh.io import export_png
from scipy.signal import savgol_filter

## Get upper bound for the plot
ub = np.max([np.max(np.abs(sieve.summarize())) for sieve in shark_results.values()])
n_summaries = len(np.unique(sieve.pattern['index']))
smoothen = True 

for shark_obj in shark_results.keys():
  sieve = shark_results[shark_obj]
  summary = sieve.summarize()
  color_pal = bokeh.palettes.viridis(n_summaries)
  figs = [figure(width=800, height=100, tools="") for _ in range(n_summaries)]
  for ii, (sig,col) in enumerate(zip(summary, color_pal)): 
    sig = savgol_filter(sig, 8, 3) if smoothen else sig
    figs[ii].line(np.linspace(0, 1, len(sieve.family)), sig, line_color=col, line_width=2.2)
    figs[ii].xaxis.visible = True
    figs[ii].yaxis.visible = False
    figs[ii].toolbar_location = None
    figs[ii].y_range = Range1d(-ub, ub)
  frame = int(re.search("sharkfixed(\\d+).obj", shark_obj).group(1))
  export_png(column(*figs), filename=f"/Users/mpiekenbrock/pbsig/animations/shark/frames_smooth/frame_{frame}.png")
  
# show(column(*figs))

## 3D plot of shark
import open3d as o3d
# from open3d.web_visualizer import draw
mesh = o3d.io.read_triangle_mesh(shark_dir+obj)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

import open3d.visualization.rendering as rendering
render = rendering.OffscreenRenderer(640, 480)
img = render.render_to_image()




from scipy.sparse.linalg import eigsh
import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(sieve.presift)
profile.add_function(sieve.project)
profile.add_function(sieve.solver.__call__)
profile.add_function(eigsh)
profile.enable_by_count()
sieve.presift(progress=True, k=10)
profile.print_stats(output_unit=1e-3, stripzeros=True)


## Visualize the sphere
from pbsig.shape import archimedean_sphere
A = archimedean_sphere(250, 5)
p = figure()
p.line(*A[:,[1,2]].T)
show(p)




