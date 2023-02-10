import numpy as np
from pbsig.linalg import * 
from pbsig.persistence import sliding_window, sw_parameters
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
output_notebook(verbose=False)
from pbsig.vis import plot_complex

f = lambda t: np.cos(t) + np.cos(3*t)
dom = np.linspace(0, 12*np.pi, 1200)

p = figure(width=400, height=200)
p.line(dom, f(dom))
show(p)


N, M = 20, 24
F = sliding_window(f, bounds=(0, 12*np.pi))
d, tau = sw_parameters(bounds=(0,12*np.pi), d=M, L=6)
#S = delaunay_complex(F(n=N, d=M, tau=tau))
X = F(n=N, d=M, tau=tau)
r = enclosing_radius(X)*0.60
S = rips_complex(X, r)
show(plot_complex(S, X[:,:2]))

print(S)
L = UpLaplacian1D(list(S.faces(2)), list(S.faces(1)))
print(len(L.pr), len(L.qr)) ## Cannot just decompress q-simplices... what if p-simplices don't partipciate in any faces? 


# plot_complex(S, pos=pca(F(n=N, d=M, tau=tau)))

scatters = []
for t in np.linspace(0.50*tau, 1.50*tau, 10):
  X_delay = F(n=N, d=M, tau=t)
  p = figure(width=150, height=150, toolbar_location=None)
  p.scatter(*pca(X_delay).T)
  scatters.append(plot_complex(S, pos=pca(X_delay), width=125, height=125))
show(row(*scatters))

## Choose a box, show its rank over vineyards 
from pbsig.persistence import * 
from pbsig.vis import plot_dgm
from scipy.spatial.distance import pdist
from splex.constructions import flag_weight
f = flag_weight(X)
K = MutableFiltration(S, f=lambda s: f(s))
# K = rips_filtration(X, r)
dgm = ph(K)
plot_dgm(dgm[1])

from pbsig.betti import mu_query
Lf = flag_weight(X, vertex_weights=np.ones(S.shape[0]))
R = np.array([-np.inf, 5, 15, np.inf])
mu_query()

all(np.ravel(L.simplices == np.array(list(K.faces(2)))))
all(np.ravel(L.faces == np.array(list(K.faces(1)))))
all(L.simplices == np.array(list(K.faces(2))))
