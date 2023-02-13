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
S = rips_complex(X, r, 2)
show(plot_complex(S, X[:,:2]))

print(S)
L = UpLaplacian1D(list(faces(S, 2)), list(faces(S, 1)))
print(len(L.pr), len(L.qr)) ## Cannot just decompress q-simplices... what if p-simplices don't partipciate in any faces? 


# plot_complex(S, pos=pca(F(n=N, d=M, tau=tau)))

scatters = []
for t in np.linspace(0.50*tau, 1.50*tau, 10):
  X_delay = F(n=N, d=M, tau=t)
  p = figure(width=150, height=150, toolbar_location=None)
  p.scatter(*pca(X_delay).T)
  scatters.append(plot_complex(S, pos=pca(X_delay), width=125, height=125))
show(row(*scatters))

## Cone the complex
S = rips_complex(X, r, 2)
S = SetComplex(S)
sv = X.shape[0] # special vertex
S.add([sv])
S.update([s + [sv] for s in faces(S, 1)])

## Choose a box, show its rank over vineyards 
from pbsig.persistence import * 
from pbsig.vis import plot_dgm
from scipy.spatial.distance import pdist
from splex.geometry import flag_weight
f = flag_weight(X)
def cone_weight(s):
  s = Simplex(s)
  if s == Simplex([sv]):
    return -np.inf
  elif sv in s:
    return np.inf
  else: 
    return f(s)
K = filtration(S, f=cone_weight)
from pbsig.persistence import ph 
dgm = ph(K)
plot_dgm(dgm[1])

## Test the multiplicity queries with the coned complex
from pbsig.betti import mu_query
R = np.array([-np.inf, 5, 15, np.inf])
mu_query(K, R=R, f=cone_weight, p=1, smoothing=(0.000000001, 1.0, 0))

R = np.array([-np.inf, 4, 15, np.inf])
mu_query(K, R=R, f=cone_weight, p=1, smoothing=(0.000000001, 1.0, 0))

## 
