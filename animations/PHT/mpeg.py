import numpy as np 
from pbsig import * 
from pbsig.linalg import * 
from pbsig.datasets import mpeg7
from pbsig.betti import *
from pbsig.pht import pht_preprocess_pc, rotate_S1
from pbsig.persistence import *
from pbsig.simplicial import cycle_graph, filtration
# import matplotlib.pyplot as plt 
from pbsig.vis import *
from bokeh.io import output_notebook

output_notebook()

# %% Load dataset 
dataset = mpeg7()

#%% Compute mu queries 
X = dataset[('bone',2)]
S = cycle_graph(X)
L = up_laplacian(S, p=0, form='lo', weight=lambda s: max((X @ np.array([1,0]))[s]) + 2.0)

## Sample a couple rectangles in the upper half-plane and plot them 
np.random.seed(1234)
R = sample_rect_halfplane(1, area=(0.10, 1.00))

## Plot vineyards 
from pbsig.persistence import ph, lower_star_filter
from bokeh.plotting import figure, show 
theta_family = np.linspace(0, 2*np.pi, 132, endpoint=False)
dgms = []
for theta in theta_family: 
  fv = X @ np.array([np.cos(theta), np.sin(theta)])
  f = lower_star_filter(fv)
  K = filtration(S, f=f)
  dgms.append(ph(K, engine="dionysus"))

from scipy.spatial.distance import pdist

theta_color = (bin_color(theta_family)*255).astype(np.uint8)
H0_vineyards = np.vstack([np.c_[np.repeat(i, len(dgm[0])), dgm[0]['birth'], dgm[0]['death']] for i, dgm in enumerate(dgms)])

rng = lambda f: max(f)-min(f)
theta_diams = np.array([rng(X @ np.array([np.cos(theta), np.sin(theta)])) for theta in theta_family])

inf_death_ind = H0_vineyards[:,2] == np.inf
H0_vineyards[:,2][inf_death_ind] =  theta_diams[H0_vineyards[:,0].astype(int)[inf_death_ind]]


theta_color = (bin_color(H0_vineyards[:,0]) *255).astype(np.uint8)
p = figure_dgm(width=400, height=400)
p.scatter(H0_vineyards[:,1], H0_vineyards[:,2], color=theta_color, size=2.5)
# show(p)


q = figure(height=400, width=400, match_aspect=True, aspect_scale=1)

c_color = (bin_color(np.linspace(0, 2*np.pi, 1500, endpoint=False))*255).astype(np.uint8)
_ = np.linspace(0, 2*np.pi, 1500, endpoint=False)
q.scatter(np.cos(_), np.sin(_), color=c_color)

# q.scatter(X[:,0], X[:,1], size=10.5, color=x_color)
q.patch(0.50*X[:,0], 0.50*X[:,1], alpha=0.40, line_width=2.5, color="blue")
q.toolbar_location = None
q.axis.visible = False
q.grid.grid_line_color = None
q.min_border_left = 0
q.min_border_right = 0
q.min_border_top = 0
q.min_border_bottom = 0

from bokeh.layouts import row
show(row(p,q))

## Plot all the diagrams, viridas coloring, as before
# from pbsig.color import bin_color
# E = np.array(list(S.faces(1)))
# vir_colors = bin_color(range(64))
# for ii, v in enumerate(uniform_S1(64)):
#   dgm = ph0_lower_star(X @ v, E, max_death='max')
#   plt.scatter(*dgm.T, color=vir_colors[ii], s=1.25)
#   i,j,k,l = R[0,:]
#   ij = np.logical_and(dgm[:,0] >= i, dgm[:,0] <= j) 
#   kl = np.logical_and(dgm[:,1] >= k, dgm[:,1] <= l) 
  #if np.any(np.logical_and(ij, kl)): print(ii)


from pbsig.betti import MuFamily

def directional_transform(X: ArrayLike):
  def _transform(theta: float):
    fv = X @ np.array([np.cos(theta), np.sin(theta)])
    return lambda s: max(fv[s])
  return _transform

## Check idempotency
X = dataset[('watch',1)]
S = cycle_graph(X)
Theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
dt = directional_transform(X)
family = [dt(theta) for theta in Theta]
sig = MuFamily(S, family, p=0)


sig.precompute(R=R[0,:])


## compare signatures manually 
Sigs = {}
for k, X in dataset.items():
  #if k[0] == 'watch':
  S = cycle_graph(X)
  dt = directional_transform(X)
  family = [dt(theta) for theta in Theta]
  Sigs[k] = MuSignature(S, family, R[0,:], form="lo")

## Precompute the singular values associated with the DT for each shape 
#keys = list(filter(lambda k: k[0] == 'watch', Sigs.keys())) # [(i,sig) for i,(k,sig) in enumerate(Sigs.items()) if k in keys]
for ii, key in enumerate(Sigs.keys()):
  Sigs[key].precompute(pp=0.30, tol=1e-4, w=1.30, normed=True)
  print(key)

s1 = Sigs[('watch',1)]()
s2 = Sigs[('watch',2)]()
s3 = Sigs[('watch',3)]()
s4 = Sigs[('watch',4)]()
s5 = Sigs[('watch',5)]()
s6 = Sigs[('watch',6)]()
s7 = Sigs[('watch',7)]()
s8 = Sigs[('watch',8)]()
plt.plot(s1); plt.plot(s2); plt.plot(s3); plt.plot(s4); plt.plot(s5);plt.plot(s6);plt.plot(s7);plt.plot(s8)

plt.plot(*dataset[('watch',1)].T)
plt.plot(*dataset[('watch',2)].T)


# %%
