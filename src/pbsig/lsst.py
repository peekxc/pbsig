import numpy as np


def low_stretch_st():
  from julia.api import Julia
  jl = Julia(compiled_modules=False) # it's unclear why this is needed, but it is
  from julia import Main
  Main.using("Laplacians")
  Main.using("LinearAlgebra")
  jl.eval("a = grid2(3)")
  A = jl.eval("a = uniformWeight(a)")
  # mst = np.array(jl.eval("mst = kruskal(a)"))
  st = np.array(jl.eval("t = akpw(a)"))
  st = np.array(jl.eval("t = randishPrim(a)"))
  st = np.array(jl.eval("t = randishKruskal(a)"))