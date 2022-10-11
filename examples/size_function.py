import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from csaps import CubicSmoothingSpline
from scipy.spatial.distance import pdist 

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
x = np.array([120, 186, 298, 456, 568, 660, 812, 998, 1140, 1340, 1429, 1533, 1644, 1743, 1864])
y = np.array([836, 525, 290, 696, 128, 189, 600, 511, 804, 254, 441, 394, 728, 830, 861])
x, y = x/1920, 1.0 - y/1080
y_cuts = 1.0 - np.array([653, 364, 242, 777])/1080
plt.scatter(x,y, s=14.50)
plt.plot(x,y)
plt.hlines(y_cuts[0], xmin=0, xmax=1, color='green')
plt.hlines(y_cuts[1], xmin=0, xmax=1, color='orange')
plt.hlines(y_cuts[2], xmin=0, xmax=1, color='purple')

from pbsig.betti import persistent_betti, lower_star_betti_sig
nv = len(x)
E = np.array(list(zip(range(nv-1), range(1, nv))))
lower_star_betti_sig([y], E, nv=nv, a=y_cuts[1], b=y_cuts[2], method="rank", keep_terms=True)


lower_star_betti_sig([y], E, nv=nv, a=y_cuts[0], b=y_cuts[2], method="rank", keep_terms=False) ## why isn't this 2?
lower_star_betti_sig([y], E, nv=nv, a=y_cuts[3], b=y_cuts[0], method="rank", keep_terms=False) ## why isn't this 2?

from pbsig.persistence import lower_star_ph_dionysus
bar0, bar1 = lower_star_ph_dionysus(y, E, [])
sum(np.logical_and(bar0[:,0] <= y_cuts[0], bar0[:,1] > y_cuts[2])) # this is one right?

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

fv = y[np.argsort(y)]
relabel = dict(enumerate(np.argsort(np.argsort(y))))

E = np.array(list(zip(range(nv-1), range(1, nv))))
E = np.array([(relabel[i], relabel[j]) for i,j in E])
E = E[np.argsort(fv[E].max(axis=1)),:]
E = np.c_[E.min(axis=1), E.max(axis=1)]

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