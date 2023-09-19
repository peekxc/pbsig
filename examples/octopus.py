import io 
import numpy as np 
from PIL import Image
from splex import * 
from pbsig.datasets import freudenthal_image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from bokeh.plotting import figure, show  
from bokeh.io import output_notebook
output_notebook()

img = Image.open("/Users/mpiekenbrock/Downloads/octopus.png")
img = np.array(img.convert("L"))
img_resized = resize(img, (img.shape[0] // 20, img.shape[1] // 20), anti_aliasing=True)
img_uint = np.array(img_resized*255, np.uint8)

## FAST 
import dionysus
filt = dionysus.fill_freudenthal(img_resized)
st = simplicial_complex([tuple(s) for s in filt if s.data <= 0.20], form="tree")

## Custom w/ embedding coordinates
X, S = freudenthal_image(img_uint, threshold=200)
st = simplicial_complex(S, "tree")
st._reindex(list(range(st.n_simplices[0])))
X = X[np.ravel(list(faces(S,0))),:]

from pbsig.vis import figure_complex
p = figure_complex(st, X, width=350, height=350)
show(p)
# np.max(np.max(st.triangles, axis=1))

## eccentricity + 0d PH ?
from pbsig.persistence import ph
from pbsig.vis import figure_dgm, g_edges
dist_to_center = np.linalg.norm(X - X.mean(axis=0), axis=1)
filt_img = filtration(st, lower_star_weight(-dist_to_center))
dgm = ph(filt_img, engine="dionysus")

## Show the 0-dimensional persistence diagram
p = figure_dgm(dgm[0])
show(p)

## Box query
box = [-0.88, -0.72, -0.52, -0.32]
box_query = lambda dgm,i,j,k,l: np.sum( \
  (dgm[0]['birth'] >= i) & (dgm[0]['birth'] <= j) & \
  (dgm[0]['death'] >= k) & (dgm[0]['death'] <= l) \
)
box_query(dgm, *box)
# (dgm[0]['birth'] >= -0.96) & (dgm[0]['birth'] <= -0.72) & \
# (dgm[0]['death'] >= -0.32) & (dgm[0]['death'] <= -0.20)

# p = figure_complex(filt_img, X[np.ravel(list(faces(S,0))),:], width=350, height=350)


filter_f = lower_star_weight(-dist_to_center)
a,b,c,d = box

## First term 
v_col = np.where(filter_f(faces(st,0)) > b, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= c, "purple", "gray")
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Second term 
v_col = np.where(filter_f(faces(st,0)) > a, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= c, "purple", "gray")
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Third term 
v_col = np.where(filter_f(faces(st,0)) > b, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= d, "purple", "gray")
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Fourth term 
v_col = np.where(filter_f(faces(st,0)) > a, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= d, "purple", "gray")
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Positive term
included_vertices_pos = (filter_f(faces(st,0)) > a).astype(int) + (filter_f(faces(st,0)) > b).astype(int)
included_edges_pos = (filter_f(faces(st,1)) <= c).astype(int) + (filter_f(faces(st,1)) <= d).astype(int)
v_col = np.array(["#8080809a", "blue", "purple"])[included_vertices_pos]
e_col = np.array(["#8080809a", "blue", "purple"])[included_edges_pos]
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Negative term
included_vertices_pos = (filter_f(faces(st,0)) > a).astype(int) + (filter_f(faces(st,0)) > b).astype(int)
included_edges_pos = (filter_f(faces(st,1)) <= c).astype(int) + (filter_f(faces(st,1)) <= d).astype(int)
v_col = np.array(["#8080809a", "orange", "red"])[included_vertices_pos]
e_col = np.array(["#8080809a", "orange", "red"])[included_edges_pos]
p = figure(width=250, height=250)
p = g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=1.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

def subset_complex(a, b):
  vertices = np.array(faces(st,0))[filter_f(faces(st,0)) > a]
  edges = np.array(faces(st,1))[filter_f(faces(st,1)) <= b]
  S = simplicial_complex(chain(iter(vertices), iter(edges)), form = "tree")
  return S



from pbsig.linalg import up_laplacian, PsdSolver, geomspace
timepoints = geomspace(1e-4, 2**8)
solver = PsdSolver()

def heat_trace(a: float, b: float):
  S_ab = simplicial_complex(subset_complex(a,b).edges, form="tree")
  L = up_laplacian(S_ab, weight=filter_f)
  ew = solver(L)
  return np.array([np.sum((1.0 - np.exp(-t*ew))) for t in timepoints])

heat_trace(b,c) - heat_trace(a,c) - heat_trace(b,d) + heat_trace(a,d)



# S_ab = subset_complex(a,b)
# np.sum(np.exp(-1.0*solver(up_laplacian(simplicial_complex(S_ab.edges)))))




## Only works for H1! 
# from pbsig.linalg import HeatKernel
# HK = HeatKernel(approx='mesh')

# HK.fit(X, S=subset_complex(b,c))
# HK.trace()
