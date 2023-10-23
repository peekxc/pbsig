# %% Imports
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from csaps import CubicSmoothingSpline
from scipy.spatial.distance import pdist 

# %% Bokeh configuration 
import bokeh
from bokeh.plotting import show, figure 
from bokeh.models import Range1d
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()

np.random.seed(170)
movement = np.random.uniform(size=50, low=-0.55, high=0.55)
position = np.cumsum(movement)+0.50

## Find parameter where we have close to 'N_EXTREMA' in the smoothed version
N_EXTREMA, lb, ub = 12, 0, 1e-3
x = np.linspace(0, 1, len(position))
def n_extrema(eps: float) -> float:
  f = CubicSmoothingSpline(x, position, smooth=1.0-eps)
  roots = f.spline.derivative(1).roots()
  #reg = 1/min(pdist(roots[:,np.newaxis], metric='cityblock'))
  return abs(len(f.spline.derivative(1).roots()) - N_EXTREMA)# + 0.01*reg
obj_cost = np.array([n_extrema(eps) for eps in np.linspace(lb, ub, 1500)])

## Fit the function
min_cost = np.min(obj_cost)
max_smooth_idx = max(np.flatnonzero((obj_cost - min_cost) == 0))
print(f"{N_EXTREMA} +/- {min_cost}")
eps = np.linspace(lb, ub, 1500)[max_smooth_idx]
f = CubicSmoothingSpline(x, position, smooth=1-eps)
dom = np.linspace(0, 1, 1500)

## Plot smoothed version + extrema 
plt.plot(dom, f(dom), linewidth=1.20)
extrema = f.spline.derivative(1).roots()
extrema = np.fromiter(filter(lambda x: x >= 0 and x <= 1.0, extrema), dtype=float)
plt.scatter(extrema, f(extrema), c='red')

## Plot function and smoothed version
# plt.plot(np.linspace(0, 1, len(position)), position)
# plt.plot(dom, f(dom))

## Test PBN on test SC
X = np.concatenate([[0], extrema, [1]], axis=0)
Y = np.concatenate([[f(0)], f(extrema), [f(1)]], axis=0)
V = np.c_[X, Y]
E = np.array(list(zip(range(V.shape[0]-1), range(1, V.shape[0]))))

plt.scatter(*V.T)
plt.plot(*V.T)




## Keynote function
x = np.array([133, 186, 298, 456, 620, 717, 826, 1002, 1149, 1340, 1440, 1526, 1641, 1816])
y = np.array([823, 525, 290, 696, 135, 319, 600, 404, 806, 254, 494, 322, 730, 865])
x, y = x/1920, 1.0 - y/1080
y_cuts = 1.0 - np.array([653, 364, 242, 777])/1080
plt.scatter(x,y, s=14.50)
plt.plot(x,y)
plt.hlines(y_cuts[0], xmin=0, xmax=1, color='green')
plt.hlines(y_cuts[1], xmin=0, xmax=1, color='orange')
plt.hlines(y_cuts[2], xmin=0, xmax=1, color='purple')

from pbsig.betti import persistent_betti, Sieve
from splex import *
nv = len(x)
E = np.array(list(zip(range(nv-1), range(1, nv))))
S, filter_f = SimplexTree(E), lower_star_filter(y)
sieve = Sieve(S, family=[filter_f], form='lo')

from pbsig.persistence import ph
K = filtration(S, filter_f)
bar0 = ph(K, engine='dionysus')[0]

from pbsig.vis import figure_dgm
show(figure_dgm(bar0))

from pbsig.linalg import spectral_rank, eigvalsh_solver
from functools import partial
machine_eps = np.sqrt(np.finfo(np.float32).eps)
inessential = ~np.isinf(bar0['death'])
b,d = bar0[inessential]['birth'], bar0[inessential]['death']

sieve.pattern = np.c_[b,d]
# sieve.pattern = np.c_[b-machine_eps, b+machine_eps, d-machine_eps, d+machine_eps]
# sieve.pattern = np.array([(0.4, 0.6, 0.6, 0.8)])

sieve.solver = eigvalsh_solver(sieve.laplacian, laplacian=True)
sieve.sift(pp=1.0, tol=0, w=0.0)
rank = partial(spectral_rank, method=0, shape=sieve.laplacian.shape)
nucl = lambda x: np.sum(np.abs(x))
sieve.summarize(rank)


from pbsig.betti import mu_query
# %% 
mu_query(S, f=lower_star_filter(y), p=0, R=np.c_[b-machine_eps, b+machine_eps, d-machine_eps, d+machine_eps][0], form='array', terms=True, smoothing=True, w=0.01, normed=True)

rect = np.c_[b-machine_eps, b+machine_eps, d-machine_eps, d+machine_eps][0]
p = figure_dgm(bar0)
p.rect(x=b[0],y=d[0],width=80*machine_eps,height=80*machine_eps, color='#ff000028')
show(p)


## From scratch: 
from pbsig.linalg import up_laplacian
D1 = boundary_matrix(K, p=1)
i,j = rect[1], rect[2] # term 0 
i,j = rect[0], rect[2] # term 1 
i,j = rect[1], rect[3] # term 2
i,j = rect[0], rect[3] # term 3 
p_weights = np.array([1 if f.value > i else 0 for f in faces(K,0)])
q_weights = np.array([1 if f.value <= j else 0 for f in faces(K,1)])

np.linalg.matrix_rank(np.diag(p_weights) @ D1 @ np.diag(q_weights))

rank(sieve.spectra[0]['eigenvalues'])
rank(sieve.spectra[1]['eigenvalues'])
rank(sieve.spectra[2]['eigenvalues'])
rank(sieve.spectra[3]['eigenvalues'])

# 11, -11, -12, 13
# face_weights = np.array([f(s) for i,s in K])
# [sum(sieve.spectra[cc]['eigenvalues']) for cc in [0,1,2,3]]
# [sum(sieve.spectra[cc]['eigenvalues']) for cc in [4,5,6,7]]

# %% Make the contour map 
from bokeh.palettes import Sunset10
f_values = np.array(list(K.indices()))
hp_bins = np.linspace(np.min(f_values), np.max(f_values), 50)
delta = np.diff(hp_bins)[0]/2
sieve.pattern = np.array([(b-delta, b+delta, d-delta, d+delta) for b,d in product(hp_bins, hp_bins) if b < d])
sieve.sift(pp=1.0, tol=0, w=1.15)

rect_centers = np.array([(b,d) for b,d in product(hp_bins, hp_bins) if b < d])
rect_tiles = np.array([(b-delta, b+delta, d-delta, d+delta) for b,d in product(hp_bins, hp_bins) if b < d])


from pbsig.linalg import sgn_approx
# reduce_f = lambda x: np.sum(np.abs(sgn_approx(x, eps=1.65)))
reduce_f = lambda x: np.sum(np.abs(x))
r_colors = bin_color(np.ravel(sieve.summarize(reduce_f)), 'viridis')
p = figure(width=300, height=300)
p.rect(x=rect_centers[:,0], y=rect_centers[:,1], width=2*delta, height=2*delta, color=r_colors)
show(p)


## Not much difference change w... should try on a larger complex that has curvature in its PL form
## i..e instead of using simple PL lines for function graph should use one that is smooth 



## One way... 
xi, yi = np.meshgrid(np.linspace(np.min(f_values), np.max(f_values),50), np.linspace(np.min(f_values), np.max(f_values),50))
z = squareform(np.ravel(sieve.summarize(nucl)))

p = figure(width=300, height=300)
levels = np.unique(np.quantile(np.ravel(z[z!=0]), q=np.linspace(0.0,1.0,10)))
# levels = np.array([0,0.25,0.5,0.75,1.0])
p.x_range = Range1d(0,1)
p.y_range = Range1d(0,1)
p.contour(xi, yi, z, levels=levels, fill_color=Sunset10, line_color="black")
p.scatter(*np.c_[b,d].T, color="green")
show(p)





#sum(np.logical_and(bar0[:,0] <= y_cuts[0], bar0[:,1] > y_cuts[2])) # this is one right?

from pbsig import plot_dgm
bar0[0,1] = max(y)
plot_dgm(bar0)
plt.vlines(y_cuts[0], ymin=0,ymax=1,linewidth=0.6, color="green")
plt.hlines(y_cuts[1], xmin=0,xmax=1,linewidth=0.6, color="orange")
plt.hlines(y_cuts[2], xmin=0,xmax=1,linewidth=0.6, color="purple")

## Size function 
from pbsig import plot_dgm
plot_dgm(bar0)
for a,b in bar0: 
  if b != np.inf:
    plt.plot([a,a], [a,b], color="black", linewidth=0.2,zorder=0)
    plt.plot([a,b], [b,b], color="black", linewidth=0.2,zorder=0)
    # plt.vlines(y_cuts[0], ymin=a,ymax=b,linewidth=0.2, color="black")

from pbsig.persistence import boundary_matrix
from pbsig.betti import persistent_betti

## Vertices labeled in domain order / edges in filtration order 
nv = len(x)
E = np.array(list(zip(range(nv-1), range(1, nv))))
# D1 = boundary_matrix({ 'vertices': list(range(nv)), 'edges': E }, p=1)
D0, D1 = boundary_matrix({ 'vertices': np.argsort(y), 'edges': E[np.argsort(y[E].max(axis=1)),:] }, p=(0,1))

## Hand-verified test cases
persistent_betti(D0, D1, i=5, j=4, summands=True) == (5, 0, 4, 3)
persistent_betti(D0, D1, i=9, j=11, summands=True) == (9, 0, 11, 4)
persistent_betti(D0, D1, i=5, j=11, summands=True) == (5, 0, 11, 8)
persistent_betti(D0, D1, i=6, j=2, summands=True) == (6, 0, 2, 1)

## Assert the lexicographically non-filtration ordered version works! 
fv = np.sort(y)
ev = np.sort(y[E].max(axis=1))
lower_star_betti_sig([y], p_simplices=E, nv=nv, a=fv[5],b=ev[4],method="rank", keep_terms=True)
lower_star_betti_sig([y], p_simplices=E, nv=nv, a=fv[9],b=ev[11],method="rank", keep_terms=True)
lower_star_betti_sig([y], p_simplices=E, nv=nv, a=fv[5],b=ev[11],method="rank", keep_terms=True)
lower_star_betti_sig([y], p_simplices=E, nv=nv, a=fv[6],b=ev[2],method="rank", keep_terms=True)

## Verify the rank evaluations are all the same
from itertools import product
for i,j in product(fv, ev):
  if i < j:
    ## This requires D1 to be in sorted order, based on y
    ii, jj = np.searchsorted(fv, i), np.searchsorted(ev,j)
    p2 = np.array(persistent_betti(D0, D1, i=ii, j=jj, summands=True))

    ## This does not 
    p1 = lower_star_betti_sig([y], p_simplices=E, nv=nv, a=i,b=j,method="rank", keep_terms=True).flatten().astype(int)
    if p1[0] == 0 and p2[0] == 0:
      continue
    assert all(abs(p1) == abs(p2)), "Failed assertion"




# fv = y[np.argsort(y)]
# relabel = dict(enumerate(np.argsort(np.argsort(y))))

# E = np.array(list(zip(range(nv-1), range(1, nv))))
# E = np.array([(relabel[i], relabel[j]) for i,j in E])
# E = E[np.argsort(fv[E].max(axis=1)),:]
# E = np.c_[E.min(axis=1), E.max(axis=1)]

D0, D1 = boundary_matrix({ 'vertices': list(range(nv)), 'edges': E }, p=(0,1))


i = np.sum(y <= y_cuts[0])
j = sum(y[E].max(axis=1) < y_cuts[2]) # inclusive! 
D1 = D1[:,np.argsort(y[E].max(axis=1))]
persistent_betti(D0, D1, i, j)


## This is right: so sum((low_entry(R1) <= i)[:(j+1)]) == rank of intersection
from pbsig.persistence import reduction_pHcol, low_entry

R0, R1, V0, V1 = reduction_pHcol(D0, D1)
i - sum((low_entry(R1) <= i)[:(j+1)]) ## Correct: 6 - 4 = 2

np.linalg.matrix_rank(D1.A) == np.linalg.matrix_rank(R1.A)

## Wrong: -11 should be -12, 9 should be 8
lower_star_betti_sig([y], E, nv=nv, a=y_cuts[0], b=y_cuts[2], method="rank", keep_terms=True)

## Correct: 12 - 8 = 4
np.linalg.matrix_rank(R1[:,:j].A) ## third term / not possible
np.linalg.matrix_rank(R1[(i+1):,:(j+1)].A) ## fourth term

sum((low_entry(R1) <= i)[:(j+1)])


## Only one that makes sense, but has rank 3! 
## Must need to permute rows...
R1[:i,:j].A

valid_edges = np.sort(y[E].max(axis=1)) <= y_cuts[2]
sum(y[low_entry(R1[:,valid_edges])] <= y_cuts[0])



## Triangle line example
x = np.array([800, 998, 1097, 933, 1097])
y = np.array([527, 617, 450, 372, 693])
x, y = x/1920, 1.0 - y/1080
y_cuts = 1.0 - np.array([587, 426])/1080
plt.scatter(x,y, s=14.50)

#K = { 'vertices': range(5), 'edges': [[0,1], [0,3], [1,3], [2,4]] }
K = { 'vertices': range(5), 'edges': [[0,1], [2,4], [0,3], [1,3]] }

## filtered edges, non-filtered vertices 
K = { 'vertices': range(5), 'edges': [[0,1], [2,4], [0,3], [1,3]] }

## filtered edges, filtered vertices 
K = { 'vertices': [4, 1, 0, 2, 3], 'edges': [[0,1], [2,4], [0,3], [1,3]] }


D0, D1 = boundary_matrix(K, p=(0,1))
D1 = D1.A % 2
E = np.array([[0,1], [2,4], [0,3], [1,3]])
lower_star_betti_sig([y], E, nv=5, a=y_cuts[0], b=y_cuts[1], method="rank", keep_terms=True)

bar0, bar1 = lower_star_ph_dionysus(y, E, [])
plot_dgm(bar0)
for a,b in bar0: 
  if b != np.inf:
    plt.plot([a,a], [a,b], color="black", linewidth=0.2,zorder=0)
    plt.plot([a,b], [b,b], color="black", linewidth=0.2,zorder=0)




  