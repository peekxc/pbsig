import numpy as np
from pbsig.linalg import * 
from pbsig.persistence import sliding_window, sw_parameters
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook(verbose=False)

f = lambda t: np.cos(t) + np.cos(3*t)
dom = np.linspace(0, 12*np.pi, 1200)

p = figure(width=400, height=200)
p.line(dom, f(dom))
show(p)


N, M = 50, 24
F = sliding_window(f, bounds=(0, 12*np.pi))
X_delay = F(n=N, d=M, L=6)

p = figure(width=250, height=250)
p.scatter(*pca(X_delay).T)
show(p)


## Choose a box, show its rank over vineyards 
from pbsig.persistence import * 
from pbsig.vis import plot_dgm
K = rips_filtration(X_delay, radius=2.0)
dgm = ph(K)
plot_dgm(dgm[1])

##
mu_query
