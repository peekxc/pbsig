import numpy as np
from scipy.spatial.distance import pdist, cdist
from itertools import * 
from pbsig.simplicial import * 
from pbsig.persistence import * 
from pbsig.vineyards import * 
import _persistence as pm

## Precompute center of each pixel + indices to index with 
def pixel_circle(n):
  vertex_pos = np.array(list(product(range(n), range(n)))) + 0.5
  center = np.array([[n/2,n/2]])
  x_dist = cdist(vertex_pos, center)/((n/2)*np.sqrt(2))

  ## Return a function
  def _circle(radius, width=0.15, e=-3.85):
    #I = np.zeros(shape=(n,n))
    return np.exp(e*(abs(x_dist - radius)/width)).reshape((n,n))
  return _circle

# from refactored.simplicial import freudenthal
n = 9
C = pixel_circle(n)
S = freudenthal(C(0))

from pbsig.vis import plot_complex
# from pbsig.simplicial import SimplicialComplex, MutableFiltration

G = np.array(list(product(range(n), range(n))))
#normalize_unit = lambda x: (x - min(x))/(max(x) - min(x))
plot_complex(S, pos=G, color = C(0.56).flatten(), palette="gray", bin_kwargs = dict(lb=0.0, ub=1.0))


## 
from splex import filtration, flag_filter, lower_star_filter
from pbsig.vis import figure_dgm
from bokeh.plotting import show

for t in np.linspace(0, 1.5, 150):
  F = filtration(S, lower_star_filter(-C(0.20).flatten()))
  dgms = ph(F, collapse=False)
  print(len(dgms[1]))

plot_complex(S, pos=G, color = 1-C(0.20).flatten(), palette="gray", bin_kwargs = dict(lb=0.0, ub=1.0))
show(figure_dgm(dgms[0]))

## Benchmark regular phcol 
from pbsig.vineyards import move_stats
OPS_PHCOL_10 = [0]
pm.reduction_stats(True)
for p in np.linspace(0, 1, num=10):
  fv = C(p).flatten() # sorted by (r,c) 
  K = MutableFiltration(S, f=lambda s: max(fv[s]))
  D = boundary_matrix(K)
  V = eye(D.shape[0])
  I = np.arange(0, D.shape[1])
  R, V = pm.phcol(D, V, I)
  OPS_PHCOL_10.extend([pm.reduction_stats(False)])

OPS_PHCOL_100 = [0]
pm.reduction_stats(True)
for p in np.linspace(0, 1, num=10):
  fv = C(p).flatten() # sorted by (r,c) 
  K = MutableFiltration(S, f=lambda s: max(fv[s]))
  D = boundary_matrix(K)
  V = eye(D.shape[0])
  I = np.arange(0, D.shape[1])
  R, V = pm.phcol(D, V, I)
  OPS_PHCOL_100.extend([pm.reduction_stats(False)])

## Vineyards baseline via phcol 
f, g = C(0.0).flatten(), C(1.0).flatten() 
K = MutableFiltration(S, f=lambda s: max(f[s]))
L = MutableFiltration(S, f=lambda s: max(g[s]))
schedule, dom = linear_homotopy(K, L)
OPS_PHCOL_FULL = [0]
pm.reduction_stats(True)
simplices = list(iter(S))
for i, x in progressbar(zip(schedule, dom), count=len(schedule)):
  fv = (1-x)*f + x*g
  index_set = [max(fv[s]) for s in simplices]
  K = MutableFiltration(zip(index_set, simplices))
  D, V = boundary_matrix(K), eye(D.shape[0]) 
  R, V = pm.phcol(D, V, list(range(len(K))))
  OPS_PHCOL_FULL.extend([pm.reduction_stats(False)])
  simplices[i], simplices[i+1] = simplices[i+1], simplices[i]



# L = K.copy()
# TR = []
# for p in np.linspace(1/100, 1.35, num=100):
#   g = C(p).flatten()
#   L.reindex(lambda s: max(g[s]))
#   schedule, _ = linear_homotopy(K, L)
#   TR.extend(schedule)
#   K.reindex(lambda s: max(g[s]))


## Benchmark vineyards
from pbsig.vineyards import vineyards_stats, transpose_rv
from pbsig.utility import progressbar
fv = C(0).flatten() # sorted by (r,c) 
K = MutableFiltration(S, f=lambda s: max(fv[s]))
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)
R = R.astype(int).tolil()
V = V.astype(int).tolil()
R = R.todense()
V = V.todense()
OPS_VINES = [vineyards_stats(reset=True)]
for p in progressbar(np.linspace(0, 1, num=100), 100):
  fv = C(p).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]), vines=True, progress=True)
  OPS_VINES.append(vineyards_stats().copy())
  #print(p)

## Make plots of the 




## TODO: solve the bottlenecks to make experimentation feasible!
import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(linear_homotopy)
profile.add_function(update_lower_star)
profile.add_function(transpose_rv)
profile.add_function(add_column)
profile.enable_by_count()
fv = C(0.20).flatten()
update_lower_star(K, R, V, f=lambda s: max(fv[s]), vines=True)
profile.print_stats(output_unit=1e-3)


## Benchmark moves 
fv = C(0).flatten() # sorted by (r,c) 
K = MutableFiltration(S, f=lambda s: max(fv[s]))
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)
R = R.astype(int).tolil()
V = V.astype(int).tolil()
# R = R.todense()
# V = V.todense()
from pbsig.vineyards import move_stats
OPS_MOVES = [move_stats(reset=True)]
for p in np.linspace(0, 1, num=20):
  fv = C(p).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]))
  #assert K.keys() do some test 
  OPS_MOVES.append(move_stats().copy())
  print(p)



#x = np.arange(n_coarse_lvls)
#x = np.arange(n_time_pts)
# [freudenthal(pixel_circle(n)(0)).shape for n in range(5,11)]
#p.line(x, ct[:,j]/n_simplices[j], color=colors.RGB(*(lc*255).astype(int)))
#p.line(x, ct[:,j], color=colors.RGB(*(lc*255).astype(int)), line_dash='dotted')






# import pickle
# with open('d_vary.pickle', 'wb') as handle:
#   pickle.dump(d_vary, handle, protocol=pickle.HIGHEST_PROTOCOL)





# import statsmodels.api as sm
# lowess = sm.nonparametric.lowess(coarsen_totals.mean(axis=1), x, frac=0.1)
# p.line(lowess[:,0], lowess[:,1], color="blue", line_width=2.5)
# 



## Bokeh plot of the cumulative number of column ops as a function of coarseness 
n_cols_total = []
for i,k in enumerate(OPS_MOVES.keys()):
  n_cols_total.append(np.array(OPS_MOVES[k]['n_cols_left']) + np.array(OPS_MOVES[k]['n_cols_right']))

from bokeh.plotting import figure, show
p = figure(title="Cumulative column operations vs coarseness", x_axis_label='Time', y_axis_label='Cum. # column cps')
lr = [p.line(np.arange(len(lt))/len(lt), lt) for lt in n_cols_total]
  














## Onto the vineyards! 
from pbsig.persistence import *
from pbsig.vineyards import *

brightness = [np.ravel(C(x)) for x in np.linspace(0, 1, 100, endpoint=False)]

plt.plot(np.array(brightness)[0,:])
plt.plot(np.array(brightness)[1,:])
plt.plot(np.array(brightness)[:,24])

np.logical_and(fv0 == 1.0, fv1 == 1.0)

fv0, fv1 = np.ravel(C(0.0)), np.ravel(C(0.40))
K0 = MutableFiltration(S, f=lambda s: max(fv0[s]))
K1 = MutableFiltration(S, f=lambda s: max(fv1[s]))


schedule_tr, dom = linear_homotopy(dict(zip(K0.values(), K0.keys())), dict(zip(K1.values(), K1.keys())), plot=True, interval=(0,0.2891037310109982))


from scipy.sparse import identity

D0 = boundary_matrix(K0)
V0 = identity(D0.shape[1]).tolil()
R0 = D0.copy().tolil()
pHcol(R0, V0)
assert is_reduced(R0)
dgm0 = generate_dgm(K0, R0)

# lower_star_ph_dionysus(fv0, E, T)

## Execute naive vineyards
n_crit = 0
it = transpose_rv(R0, V0, schedule_tr)
F = list(K0.values())
for cc, (status,xval) in enumerate(zip(it, dom)):
  i = schedule_tr[cc]
  tau, sigma = F[i], F[i+1]
  assert is_reduced(R0)
  assert not(tau in sigma)
  print(f"{cc}: status={status}, nnz={R0.count_nonzero()}, n pivots={sum(low_entry(R0) != -1)}, pair: {(F[i], F[i+1])}")
  F[i], F[i+1] = F[i+1], F[i]
  fv = (1-xval)*fv0 + xval*fv1
  if status in [2,5,7]:
    n_crit += 1 
    dgm0_dn = lower_star_ph_dionysus(fv, E, T)
    dgm0_vi = generate_dgm(MutableFiltration(S, f=lambda s: max(fv[s])), R0)
    assert len(dgm0_vi) == (len(dgm0_dn[0]) + len(dgm0_dn[1]))


## Moves (global)
D = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
R = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]])
V = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
D = scipy.sparse.bmat([[None, D], [np.zeros((3,3)), None]]).tolil()
R = scipy.sparse.bmat([[None, R], [np.zeros((3,3)), None]]).tolil()
V = scipy.sparse.bmat([[np.eye(3), None], [None, V]]).tolil()
assert np.allclose(R.todense(), (D @ V).todense() % 2)
R, V = R.astype(int), V.astype(int)

## Move right (global)
#move_right(R, V, 3, 5)
#assert np.allclose(R.todense(), (permute_cylic_pure(D, 3, 5) @ V).todense() % 2)

## TODO: 
## 1. ensure face poset is respected (done; not needed!)
## 2. Implement greedy heuristic (done)

print(R)
print(V)
move_right(R, V, 0, 2)
print(R)
print(V)
np.allclose(R, permute_cylic_pure(D, i,j,"cols") @ V % 2 )
