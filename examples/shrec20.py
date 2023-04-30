from scipy.spatial.distance import pdist,squareform
from pbsig.datasets import shrec20 
from pbsig.shape import archimedean_sphere
from splex import * 

X, S = shrec20()
S2_pts = archimedean_sphere(n = 100, nr = 5)
diam_X = np.max(pdist(X))

fv = np.array(X) @ np.array([0,0,1])
fv += abs(np.min(fv))
f = lower_star_weight(fv)

from pbsig.linalg import up_laplacian, pseudoinverse, LaplacianOperator
op = up_laplacian(S, p=1, form='lo', weight=f, normed=True)
L = LaplacianOperator(S, op, normalized=True)

L.update_scalar_product(f)

from scipy.sparse.linalg import eigsh
from pbsig.utility import progressbar

spri = []
for pt in progressbar(S2_pts):
  fv = np.array(X) @ np.array(pt)
  fv += abs(np.min(fv))
  f = lower_star_weight(fv)
  L.update_scalar_product(f)
  spri.append(eigsh(op, k=20, which='LM', tol=1e-5, return_eigenvectors=False))

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

p = figure()
p.line(np.arange(len(spri)), [sum(v) for v in spri])
show(p)