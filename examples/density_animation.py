from bokeh.io import output_notebook, export_png
# from bokeh.models import 
output_notebook()

# %% Load data 
from pbsig.datasets import noisy_circle
from splex.geometry import delaunay_complex
np.random.seed(1234)
X = noisy_circle(80, n_noise=30, perturb=0.15)
S = delaunay_complex(X)

# %% Create filtering function
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist

def codensity(bw: float):
  x_density = gaussian_kde(X.T, bw_method=bw).evaluate(X.T)
  x_codensity = max(x_density) - x_density
  return x_codensity

## TODO: make vectorized ! and don't assume vertices are labeled 0...(n-1)
def lower_star_weight(x: ArrayLike) -> Callable:
  def _weight(s: SimplexConvertible) -> float:
    return max(x[s])
  return _weight

from splex import *
from pbsig.persistence import ph
from pbsig.vineyards import update_lower_star
x_codensity = codensity(0.05)
K = filtration(S, f = lambda s: max(x_codensity[s]))
R,V = ph(K, p=None, output="RV", engine="python", validate=False)
B = boundary_matrix(K)
## Make a spy plot 
n = R.shape[0]
# D = np.zeros((n,n), dtype=np.uint32)
# D_view = D.view(dtype=np.int8).reshape((n, n, 4))
# for i,j in zip(*R.nonzero()):
#   D_view[i,j,0] = 255
#   D_view[i,j,1] = 255
#   D_view[i,j,2] = 255
#   D_view[i,j,3] = 255
fig_kw = dict(width=200, height=200) #  | kwargs
p = figure(**fig_kw)
rect_coords = np.array([(j,i,4,4) for i,j in zip(*R.nonzero())])
p.rect(*rect_coords.T)
#   p.rect(i,j,width=1, height=1)
p.x_range.range_padding = p.y_range.range_padding = 0
p.y_range.flipped = True
p.x_range.flipped = False
show(p)
# p.y_range = Range1d(0,-10)
# p.x_range = Range1d(0,10)

# p.background_fill_color = "#000000"
# p.image_rgba(image=[np.flipud(D)], x=0, y=0, dw=10, dh=10, dilate=True, coor='blue')

# update_lower_star(K, R, V, lower_star_weight(codensity(0.10)), vines=True, progress=True)
# export_png(plot, filename="plot.png")