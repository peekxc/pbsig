# %% [markdown] 
# title: Classification using the Persistent Homology transform
# author: Matt Piekenbrock
# date: 09/14/23

# %% Imports
#| echo: false
import numpy as np 
from pbsig import * 
from pbsig.linalg import * 
from pbsig.datasets import mpeg7
from pbsig.betti import *
from pbsig.persistence import *
from pbsig.simplicial import cycle_graph
from splex import *
from scipy.spatial.distance import pdist, squareform

# %% Load dataset
# NOTE: PHT preprocessing centers and scales each S to *approximately* the box [-1,1] x [-1,1]
dataset = mpeg7(simplify=200)
X_, y_ = mpeg7(simplify=200, return_X_y=True)
print(dataset.keys())

# %% Construct an interpolated operator 
from pbsig.interpolate import ParameterizedLaplacian
from pbsig.pht import directional_transform

## Make a parameterized laplacian over a *fixed* complex
S = cycle_graph(200-1)
LP = ParameterizedLaplacian(S, p=0)

## Choose a data set to parameterize a directional transform over
X = dataset[('bat',1)]
F = directional_transform(X, dv=16, normalize=True, nonnegative=1.0)
LP.interpolate_family(F, interval=[0, 1])

# %% Try the heat trace?
from pbsig.vis import figure_complex
import bokeh
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
output_notebook()

p = figure_complex(S, pos = X, width = 250, height = 250, vertex_size=2, edge_line_width=5.5, edge_line_color="red")
show(p)
# np.array([HK.trace(LP.laplace(t), p=0.5, gram=False, method='eigenvalue') for t in np.linspace(0,1,32)])


# %% Try the heat trace?
from pbsig.linalg import HeatKernel
from pbsig.utility import progressbar
HK = HeatKernel(approx="mesh")

hk_traces = []
for data_key, X in progressbar(dataset.items()):
  HK.fit(X=X, S=S, use_triangles=False)
  hk_traces.append(HK.trace())

## Plot the heat traces for varying t 
from pbsig.color import bin_color, colors_to_hex
class_colors = bin_color(np.unique(y_), 'turbo')
p = figure(width=250, height=200)
for label, tr in zip(y_, hk_traces):
  line_color = tuple((class_colors[label][:3]*255).astype(int))
  p.line(np.arange(len(tr)), tr, line_color=line_color)
show(p)


















from numpy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh
from imate import trace, schatten, toeplitz
# A = toeplitz(20, 19, size=20, gram=True, format='csr')
# all((A == A.T).data)

# ew = eigvalsh(L.todense())
# n = L.shape[0]
# np.sum(np.sqrt(ew))/n ## nuclear norm of B where A = B^T B 

# p = 0.5
# # print((sum(ew) * (n**(-1/p))))
# print(sum(ew**p))
# print(trace(L, p=p, gram=False, method='eigenvalue')) # this one, but slq doesn't work
# trace(L, p=p, gram=True, method="eigenvalue") # close

schatten_norms = np.array([schatten(LP.laplace(t), p=1.0, method="exact") for t in np.linspace(0,1,32)])




A = L
sns.set(font_scale=1.15)
sns.set_style("white")
sns.set_style("ticks")
p = np.linspace(-0.01, 2.5, 201)
norm = np.zeros_like(p)
for i in range(p.size):
  norm[i] = schatten(A, p=p[i], method='eigenvalue')
norm0 = schatten(A, p=0, method='eigenvalue')
plt.semilogy(p, norm, color='black')
plt.semilogy(0, norm0, 'o', color='black')
plt.xlim([p[0]-0.5, p[-1]])
plt.ylim([0, 1e1])
plt.xlabel('$p$')
plt.ylabel('$\Vert \mathbf{A} \Vert_p$')
plt.title(r'Schatten Norm')
plt.show()


p = figure()
p.line(np.linspace(0,1,32), schatten_norms)
p.scatter(np.linspace(0,1,32), schatten_norms)
show(p)

# %% Show the graph filtering for different matrix functions across classes



# %%
# from pbsig.apparent_pairs import apparent_pairs
# from primate import slq
from pbsig.pht import directional_transform
from pbsig.betti import sample_rect_halfplane
dt = directional_transform(X, dv = 16)


rects = sample_rect_halfplane(5, lb = 0.0, ub = 1.001*max(pdist(X)), disjoint=True)



L = up_laplacian(S, p=0, form='array', weight=f)


# for f in dt:
#   L = up_laplacian(S, p=0, form='array', weight=f)
#   print(slq(L, matrix_function="heat", max_num_samples=50, error_atol=1e-10, error_rtol=1e-8, num_threads=6))



# %%
from pbsig.betti import Sieve
sieve = Sieve(S, dt)

from scipy.sparse.linalg import eigsh
eigsh(L, k=100, return_eigenvectors=False)


lower_star_weight((X @ np.array([0,1])) + radius)

# from scipy.sparse import eye
# from scipy.sparse.linalg import aslinearoperator
# cI = aslinearoperator(0.01*eye(L.shape[0]))

## Sample a couple rectangles in the upper half-plane and plot them 
R = sample_rect_halfplane(1, area=(0.10, 1.00))

## Plot all the diagrams, viridas coloring, as before
from pbsig.color import bin_color
E = np.array(list(S.faces(1)))
vir_colors = bin_color(range(64))
for ii, v in enumerate(uniform_S1(64)):
  dgm = ph0_lower_star(X @ v, E, max_death='max')
  plt.scatter(*dgm.T, color=vir_colors[ii], s=1.25)
  i,j,k,l = R[0,:]
  ij = np.logical_and(dgm[:,0] >= i, dgm[:,0] <= j) 
  kl = np.logical_and(dgm[:,1] >= k, dgm[:,1] <= l) 

