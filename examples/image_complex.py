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

n = 9
C = pixel_circle(n)
S = freudenthal(C(0))

from pbsig.vis import plot_complex
from pbsig.simplicial import freudenthal, SimplicialComplex, MutableFiltration

G = np.array(list(product(range(n), range(n))))
#normalize_unit = lambda x: (x - min(x))/(max(x) - min(x))
plot_complex(S, pos=G, color = C(0.56).flatten(), palette="gray", bin_kwargs = dict(lb=0.0, ub=1.0))


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



## Compare the cost by varying the number of moves (d) as a function of the size (n)
np.random.seed(1234)
OPS_MOVES_ND = { }
for n in range(5,11,1): ## grid sizes
  C = pixel_circle(n)
  S = freudenthal(C(0))

  def init_rv():
    fv = C(0).flatten() # sorted by (r,c) 
    K = MutableFiltration(S, f=lambda s: max(fv[s]))
    D, V = boundary_matrix(K), eye(len(K))
    R, V = pm.phcol(D, V, range(len(K)))
    R, V = R.astype(int).tolil(), V.astype(int).tolil()
    return K, R, V

  n_time_pts = 10
  n_coarse_lvls = 5
  # n_retrials = 10
  for cc, cf in enumerate(np.linspace(0, 1, n_coarse_lvls)):
    K,R,V = init_rv()
    stats = { k : [] for k,x in move_stats(reset=True).items() }
    for radius in np.linspace(0, 1, num=n_time_pts):
      fv = C(radius).flatten()
      update_lower_star(K, R, V, f=lambda s: max(fv[s]), coarsen=cf, method="greedy")
      for k,v in move_stats().items():
        stats[k].extend([v])
    OPS_MOVES_ND[(cc,n)] = stats
  #print(cc)
  print(n)

# import pickle
# with open('OPS_MODES_ND.pickle', 'wb') as handle:
#   pickle.dump(OPS_MOVES_ND, handle, protocol=pickle.HIGHEST_PROTOCOL)

calculate_total = lambda ms: np.array(ms["n_cols_left"]) + np.array(ms["n_cols_right"])
ct0 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 5]).T
ct1 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 6]).T
ct2 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 7]).T
ct3 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 8]).T
ct4 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 9]).T
ct5 = np.vstack([calculate_total(OPS_MOVES_ND[(cc,n)]) for cc,n in OPS_MOVES_ND.keys() if n == 10]).T

n_simplices = [sum(freudenthal(pixel_circle(n)(0)).shape) for n in range(5,11)]
from bokeh import colors
p = figure(width=300, height=300)
#x = np.arange(n_coarse_lvls)
x = np.arange(n_time_pts)
CT = [ct0,ct1,ct2,ct3,ct4,ct5]
# for ct in CT:p.scatter(x, ct.max(axis=0))
line_colors = bin_color(range(len(CT)+1), color_pal="inferno")
for ct, lc in zip(CT, line_colors):
  #for j in range(ct.shape[1]):
  j = 4
  #p.line(x, ct[:,j]/n_simplices[j], color=colors.RGB(*(lc*255).astype(int)))
  p.line(x, ct[:,j], color=colors.RGB(*(lc*255).astype(int)), line_dash='dotted')
show(p)
# [freudenthal(pixel_circle(n)(0)).shape for n in range(5,11)]








## Compare the cost by varying the number of moves (d) as a function of the size (n)
from pbsig.vineyards import *
np.random.seed(1234)
n = 9
C = pixel_circle(n)
S = freudenthal(C(0))
n_time_pts = 25

def init_rv():
  fv = C(0).flatten() # sorted by (r,c) 
  K = MutableFiltration(S, f=lambda s: max(fv[s]))
  D, V = boundary_matrix(K), eye(len(K))
  R, V = pm.phcol(D, V, range(len(K)))
  R, V = R.astype(int).tolil(), V.astype(int).tolil()
  return K, R, V

OPS_MOVES = { }
for cc, cf in enumerate(np.linspace(0, 1, 6)):
  for i in range(10): ## repeat to average
    K,R,V = init_rv()
    stats = { k : [] for k,x in move_stats(reset=True).items() }
    for radius in np.linspace(0, 1, num=n_time_pts):
      fv = C(radius).flatten()
      update_lower_star(K, R, V, f=lambda s: max(fv[s]), coarsen=cf, method="random")
      for k,v in move_stats().items():
        stats[k].extend([v])
    OPS_MOVES[(cc,i)] = stats
  print(cc)

## Greedy strategy 
K,R,V = init_rv()
greedy_stats = { k : [] for k,x in move_stats(reset=True).items() }
for radius in np.linspace(0, 1, num=n_time_pts):
  fv = C(radius).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]), method="greedy")
  for k,v in move_stats().items():
    greedy_stats[k].extend([v])

## Naive strategy
K,R,V = init_rv()
naive_stats = { k : [] for k,x in move_stats(reset=True).items() }
for radius in np.linspace(0, 1, num=n_time_pts):
  fv = C(radius).flatten()
  update_lower_star(K, R, V, f=lambda s: max(fv[s]), method="naive")
  for k,v in move_stats().items():
    naive_stats[k].extend([v])

# import pickle
# with open('coarsening_data.pickle', 'wb') as handle:
#   pickle.dump(OPS_MOVES, handle, protocol=pickle.HIGHEST_PROTOCOL)



## Load the first figure
import pickle
# with open('coarsening_data.pickle', 'wb') as handle:
#   pickle.dump(OPS_MOVES, handle, protocol=pickle.HIGHEST_PROTOCOL)
OPS_MOVES = pickle.load(open('coarsening_data.pickle', 'rb'))

calculate_total = lambda ms: np.array(ms["n_cols_left"]) + np.array(ms["n_cols_right"])
ct0 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 0]).T
ct1 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 1]).T
ct2 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 2]).T
ct3 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 3]).T
ct4 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 4]).T
ct5 = np.vstack([calculate_total(OPS_MOVES[(cc,i)]) for cc,i in OPS_MOVES.keys() if cc == 5]).T


from bokeh import colors
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, BasicTickFormatter, Band
CT = [ct0,ct1,ct2,ct5] #,ct3,ct4

from pbsig.color import bin_color
#ct_colors = ["blue", "green", "yellow", "orange", "red", "black"]
ct_colors = bin_color(range(5), color_pal="inferno")
p = figure(
  title="Move schedule totals on grayscale image data (9x9)", 
  x_axis_label='Time', y_axis_label='Cumulative number of column operations',
  x_range=(0,1), width=500, height=350, 
  tooltips=None
)
# p.yaxis.formatter.use_scientific = True 
p.yaxis.formatter = BasicTickFormatter(use_scientific=True)
p.output_backend = "svg"
naive_y = calculate_total(naive_stats)
p.line(x, naive_y, color="red", line_width=2.5, line_dash="dashed", legend_label="0 (naive)")
for i, (ct, col) in enumerate(zip(CT, ct_colors)):
  x = np.arange(ct.shape[0])/ct.shape[0]
  col = colors.RGB(*(col*255).astype(int))
  legend_text = f"{i}"
  legend_text = f"{i} (none)" if i == 0 else legend_text
  legend_text = f"{i} (33%)" if i == 1 else legend_text
  legend_text = f"{i} (66%)" if i == 2 else legend_text
  legend_text = f"{i} (maximum)" if i == 3 else legend_text
  p.line(x, ct.mean(axis=1), color=col, line_width=2.25, legend_label=legend_text)
  band_source = ColumnDataSource({
    'base': x,
    'lower': ct.min(axis=1),
    'upper': ct.max(axis=1)
  })
  band = Band(base="base", lower='lower', upper='upper', source=band_source, fill_alpha=0.10, fill_color=col)
  p.add_layout(band)
  # display legend in top left corner (default is top right corner)
p.legend.location = "top_left"
p.legend.title = "Coarsening level"
p.legend.border_line_width = 1
p.legend.border_line_color = "black"
# p.legend.label_text_line_height = 0.2
p.legend.label_height = 10
p.legend.glyph_height = 10
p.legend.spacing = 2
greedy_y = calculate_total(greedy_stats)
p.line(x, greedy_y, color="orange", line_width=2.5, line_dash="dotdash", legend_label="3 (greedy)")
show(p)

from bokeh.io import export_svg, export_png
export_svg(p, filename="coarsening_grayscale.svg")
export_png(p, filename="coarsening_grayscale.png" )



# change appearance of legend text
p.legend.label_text_font = "times"
p.legend.label_text_font_style = "italic"
p.legend.label_text_color = "navy"

# change border and background of legend
p.legend.border_line_width = 3
p.legend.border_line_color = "navy"
p.legend.border_line_alpha = 0.8
p.legend.background_fill_color = "navy"
p.legend.background_fill_alpha = 0.2

show(p)

p = figure(x_range=(0,1), y_range=(0, 7000))
source = ColumnDataSource({
  'base':x,
  'lower':ct.min(axis=1),
  'upper':ct.max(axis=1)
})
band = Band(base='base', lower='lower', upper='upper', source=source, fill_alpha=0.5)
p.add_layout(band)
show(p)





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
