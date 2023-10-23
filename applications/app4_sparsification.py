import numpy as np
from pbsig.datasets import noisy_circle
from pbsig.vis import figure_complex
from simplextree import SimplexTree
from pbsig.shape import landmarks
from scipy.spatial.distance import pdist, cdist, squareform
from itertools import combinations
from functools import partial
from splex import rips_complex, enclosing_radius

X = noisy_circle()
dx = pdist(X)
landmark_ind, ins_radii = landmarks(dx, X.shape[0])
insert_map = np.zeros(len(X))
insert_map[landmark_ind] = ins_radii
enc_radius = enclosing_radius(X)

## TODO: subtract offset constant
# a = min(p_filter(t=0.0, p=1))
# b = min(p_filter(t=1.0, p=1))
shift = lambda eps: (1-eps)*a + eps*b

from functools import partial 
def vertex_weight(v: int, eps: float, alpha: float):
  ir = insert_map[v]
  if alpha <= (ir / eps):
    return 0.0
  elif alpha < (ir / (eps * (1 - eps))):
    return (alpha - ir / eps) - shift(eps)
  else:
    return (eps * alpha) - shift(eps)

def relaxed_diam(eps: float):
  from splex import flag_filter
  vertex_weights = np.array([vertex_weight(i, eps, enc_radius) for i in range(len(X))])
  edge_weights = dx + np.array([wi + wj for wi, wj in combinations(vertex_weights, 2)])
  return flag_filter(edge_weights, vertex_weights)

from pbsig.interpolate import ParameterizedFilter
K = rips_complex(X, p=2, radius = enc_radius)
eps_values = np.linspace(0, 1, 50)[1:-1]
filter_funcs = list((relaxed_diam(eps) for eps in eps_values))
p_filter = ParameterizedFilter(S = K, family = filter_funcs)
# p_filter.interpolate()

from pbsig.vis import bin_color, figure_dgm
from bokeh.io import output_notebook
output_notebook()
dgms = [ph(filtration(K, filter_f), engine="dionysus") for filter_f in p_filter]

from  bokeh.plotting import show
p = figure_dgm(width=300, height=300)
vine_colors = bin_color(np.arange(len(dgms)), "viridis")
for dgm, vc in zip(dgms, vine_colors):
  d = dgm[1]
  print(sum(d['death'] == np.inf))
  valid_pairs = d['death'] != np.inf
  pt_color = (vc*255).astype(np.uint8)
  pt_color = np.array([pt_color]*sum(valid_pairs))
  p.scatter(d['birth'][valid_pairs], d['death'][valid_pairs], color=pt_color)
show(p)

# from pbsig.persistence import generate_dgm, validate_decomp, boundary_matrix
# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(ph)
# profile.add_function(boundary_matrix)
# profile.add_function(generate_dgm)
# profile.add_function(validate_decomp)
# profile.enable_by_count()
# ph(K_filt, engine="cpp") # its all in the slow _boundary call in splex 
# profile.print_stats(output_unit=1e-3, stripzeros=True)



# p_filter(t=0.0)

## Make the persistence diagram for varying eps



# from splex import * 
# from combin import rank_to_comb, comb_to_rank
# from hirola import HashTable

# ## Make a great vectorized dict replacement to enable fast filter function
# rd = np.dtype([('r', np.uint64), ('k', np.uint8)])
# index_table = HashTable(len(K) * 1.25, rd)
# v_ind = np.fromiter(zip(comb_to_rank(faces(K, 0)), np.repeat(1, card(K,0))), dtype=rd)
# e_ind = np.fromiter(zip(comb_to_rank(faces(K, 1)), np.repeat(2, card(K,1))), dtype=rd)
# t_ind = np.fromiter(zip(comb_to_rank(faces(K, 2)), np.repeat(3, card(K,2))), dtype=rd)

# index_table.add(v_ind)
# index_table.add(e_ind)
# index_table.add(t_ind)

# index_table[t_ind]





from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from bokeh.models import * 
output_notebook()

edge_values = np.array([p_filter(t=eps, p=1) for eps in eps_values])

from pbsig.vis import figure_hist
fig = figure(width=350, height=200)
for i in range(card(K, 1)):
  fig.line(eps_values, edge_values[:,i])
show(fig)

show(figure_hist(*np.histogram(edge_values[0,:])))
show(figure_hist(*np.histogram(edge_values[4,:])))
show(figure_hist(*np.histogram(edge_values[17,:])))


# %% Sheehy's 
def edge_birth_time(e: tuple, eps: float):
  i, j = e
  min_insert = min(insert_map[i], insert_map[j])
  if (dx[i,j] <= (2*min_insert*(1+eps))/eps):
    return dx[i,j] / 2
  elif (dx[i,j] <= ((insert_map[i] + insert_map[j])*(1 + eps))/eps):
    return dx[i,j] - min_insert * (1+eps) / eps
  else: 
    return np.inf
def simplex_birth_time(sigma: tuple, eps: float):
  if len(sigma) == 1: 
    return 0.0
  elif len(sigma) == 2: 
    return edge_birth_time(sigma, eps)
  else:
    ew = partial(edge_birth_time, eps=eps)
    e1, e2, e3 = map(ew, faces(sigma, 1))
    max_weight = np.max([e1,e2,e3]) 
    # min_birth = np.min(insert_map[sigma] * (1 + eps)**2 / eps)
    # return max_weight if max_weight <= min_birth else np.inf
    return max_weight

from pbsig.persistence import ph
from splex import generic_filter, flag_filter, filtration

# F = filtration(K, f=f_sparse)
f_sparse = generic_filter(simplex_birth_time)
X_weight = f_sparse(combinations(range(len(X)), 2))

(len(X_weight) - np.sum(X_weight == np.inf))/len(X_weight)

## TODO: vary epsilon in 0 < eps < 1 for all edges and triangles, verify discontinuous behavior 

dgm = ph(F)

dgm[1]
