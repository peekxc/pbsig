import numpy as np
import trimesh
from splex import *
from pbsig.betti import Sieve
from pbsig.pht import parameterize_dt
from pbsig.linalg import *

mesh = trimesh.load("/Users/mpiekenbrock/Downloads/Shark002.blend/frames/sharkfixed46.obj", force='mesh')
mesh.show()

## Need filter functions to by ufuncs 
F = np.array(mesh.faces)
F = np.array([np.sort(p) for p in F])
F = F[np.lexsort(np.rot90(F)),:]

## Turn mesh into complex and parameterize a directional transform
S = simplicial_complex(F)
DT = parameterize_dt(np.array(mesh.vertices), 10, nonnegative=False)

## Create the sieve with a random pattern 
sieve = Sieve(S, DT)
sieve.randomize_pattern(2)

## Choose a sensible tolerance and sift through the family
sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
sieve.sift(progress=True, k=10) # most expensive line
# sieve.sift(pp=0.05)

## Collect the results 
sieve.summarize()




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


sieve.presift()


sieve._pattern

from pbsig.shape import archimedean_sphere
A = archimedean_sphere(250, 5)
import bokeh
from bokeh.plotting import figure, show 
from bokeh.io import output_notebook
output_notebook()
p = figure()
p.line(*A[:,[1,2]].T)
show(p)




