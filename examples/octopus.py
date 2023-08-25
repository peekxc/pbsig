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
dist_to_center = np.linalg.norm(X - X.mean(axis=0), axis=1)
filt_img = filtration(st, lower_star_weight(-dist_to_center))

from pbsig.persistence import ph
dgm = ph(filt_img, engine="dionysus")

from pbsig.vis import figure_dgm, g_edges
p = figure_dgm(dgm[0])
show(p)

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
g_edges(p, edges=st.edges, pos=X, color=e_col, line_width=2.5)
p.scatter(*X.T, color=v_col, size=2.5)
show(p)

## Second term 
v_col = np.where(filter_f(faces(st,0)) > a, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= c, "purple", "gray")
p = figure(width=250, height=250)
g_edges(p, edges=st.edges, pos=X, color=e_col)
p.scatter(*X.T, color=v_col)
show(p)

## Third term 
v_col = np.where(filter_f(faces(st,0)) > b, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= d, "purple", "gray")
p = figure(width=250, height=250)
g_edges(p, edges=st.edges, pos=X, color=e_col)
p.scatter(*X.T, color=v_col)
show(p)

## Fourth term 
v_col = np.where(filter_f(faces(st,0)) > a, "blue", "gray")
e_col = np.where(filter_f(faces(st,1)) <= d, "purple", "gray")
p = figure(width=250, height=250)
g_edges(p, edges=st.edges, pos=X, color=e_col)
p.scatter(*X.T, color=v_col)
show(p)