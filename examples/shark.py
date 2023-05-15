import numpy as np
import trimesh
from splex import *
from pbsig.betti import Sieve
from pbsig.pht import parameterize_dt

mesh = trimesh.load("/Users/mpiekenbrock/Downloads/Shark002.blend/frames/sharkfixed46.obj", force='mesh')
mesh.show()

## Need filter functions to by ufuncs 
F = np.array(mesh.faces)
F = np.array([np.sort(p) for p in F])
F = F[np.lexsort(np.rot90(F)),:]

## Turn mesh into complex and parameterize a directional transform
S = simplicial_complex(F)
DT = parameterize_dt(np.array(mesh.vertices), 10, nonnegative=False)

## Create the sieve
sieve = Sieve(S, DT)
sieve.randomize_pattern(2)
sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)


import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(sieve.presift)
profile.add_function(sieve.project)
profile.add_function(sieve.solver.__call__)
profile.enable_by_count()
sieve.presift(progress=True, k=10)
profile.print_stats(output_unit=1e-3, stripzeros=True)


## Benchmark construction time 
import timeit
timeit.timeit(lambda: simplicial_complex(F, form="tree"), number=10)
timeit.timeit(lambda: simplicial_complex(F, form="set"), number=10)
timeit.timeit(lambda: simplicial_complex(F, form="rank"), number=10)

S1, S2, S3 =  simplicial_complex(F, form="set"),  simplicial_complex(F, form="rank"),  simplicial_complex(F, form="tree")
timeit.timeit(lambda: [max(s) for s in S1], number=10)
timeit.timeit(lambda: [max(s) for s in S2], number=10)
timeit.timeit(lambda: [max(s) for s in S3], number=10)

p = sieve.figure_pattern()
show(p)

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




