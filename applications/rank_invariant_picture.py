# %% Imports
from typing import *
import numpy as np
import splex as sx
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from pbsig.betti import SpectralRankInvariant
from pbsig.datasets import noisy_circle
from pbsig.interpolate import ParameterizedFilter
from pbsig.persistence import ph
from pbsig.vis import *
from scipy.interpolate import BSpline, splrep
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from pbsig.datasets import PatchData
from pbsig.vis import figure_complex, show
from pbsig.utility import progressbar
output_notebook()

# %% 
from pbsig.simplicial import path_graph
from pbsig.datasets import random_function
np.random.seed(1247)
f, f_p = random_function(n_extrema=12, n_pts=1500, walk_distance=0.005, plot=True)
S = path_graph(150)
sublevel_filter = sx.lower_star_filter(f(np.linspace(0,1,150)))

# %% Show the persistence diagram
from pbsig.persistence import ph
dgms = ph(sx.filtration(S, sublevel_filter))
show(figure_dgm(dgms[0]))

# %% Draw the size function 
pp = figure_dgm(dgms[0])
for a,b in dgms[0]:
  if b != np.inf:
    pp.patch([a,a,b], [a,b,b], color='red', fill_alpha=0.20)
pp = figure_dgm(dgms[0], figure=pp)
show(pp)

# %% 
from bokeh.layouts import row, column 
p1 = f_p 
p2 = figure_dgm(dgms[0])
p1.toolbar_location = None
p2.toolbar_location = None 
p1.title = "Real-valued function"
p2.height = p1.height = 300 
show(row(p1, p2))


# %% Color the cells
from shapely import Polygon, GeometryCollection, polygonize, polygonize_full, get_geometry, simplify, get_parts, unary_union, union, union_all
from shapely import LineString, multipolygons, segmentize, boundary, make_valid, envelope
from shapely.ops import split
# triangles = [Polygon(np.c_[[a,a,b], [a,b,b]]) for a,b in dgms[0] if b != np.inf]
# rank_areas = simplify(polygonize(triangles), 0.00001, preserve_topology=False)

## This seems to be the magical code to partition a set of lines
from more_itertools import collapse, unique_everseen
lines = list(collapse([(LineString([[a,a], [a,b]]),LineString([[a,b], [b,b]]), LineString([[a,a], [b,b]])) for a,b in dgms[0] if b != np.inf]))
lines = list(unique_everseen(lines))
multi_polygon = multipolygons(get_parts(polygonize(lines)))
multi_polygon = make_valid(multi_polygon)
main_triangle = boundary(get_geometry(multi_polygon,0))
inner_lines = get_geometry(multi_polygon,1)
all_lines = union(main_triangle, inner_lines)
poly_areas = get_parts(polygonize(get_parts(all_lines)))

## Get the polygon indices in descending order of area
desc_inds = np.argsort(-np.array([po.area for po in poly_areas]))

## Get the coordinates of the upper-left corners
leftupper_corner = lambda p: (p[0], p[3])
np.array([leftupper_corner(envelope(pa).bounds) for pa in poly_areas])

poly_colors = bin_color(np.arange(len(poly_areas)), 'Category10')

from bokeh.models import Label
labels = { 3 : 2, 4 : 3, 2 : 3, 5 : 4 }
pp = figure_dgm(dgms[0])
for i, pi in enumerate(desc_inds):
  px, py = poly_areas[pi].exterior.coords.xy
  pp.patch(px, py, fill_color=rgb_to_hex(poly_colors[i]*255), fill_alpha=0.75, line_alpha=0.0)
  cx, cy = poly_areas[pi].centroid.coords.xy
  # pp.scatter(cx,cy)
  if poly_areas[pi].area > 1:
    x = np.take(cx,0)-0.025
    x = x - 0.50 if pi == 3 else x 
    x = x - 0.25 if pi == 5 else x
    y = np.take(cy,0) if pi != 4 else np.take(cy,0) - 0.25
    print(f"{pi}, {poly_areas[pi].area}")
    pp.add_layout(Label(x=x, y=y, text=str(labels[pi]), text_font_size='20px', text_font_style='bold'))
  # pp.text(x=cx, y=cy, text="Hello", text_color='black')

pp = figure_dgm(dgms[0], figure=pp)
pp.toolbar_location = None
pp.title = "Size Function"
show(pp)

# %% 
show(row(p1, pp))

# %% Compute the spectral rank on a finite grid
from pbsig.betti import betti_query, BettiQuery
from pbsig.linalg import spectral_rank
from pbsig.csgraph import up_laplacian

## Construct the grid points based on the locations of the point in the diagram
n_grid_pts = 30
b, d = dgms[0]['birth'], dgms[0]['death']
min_diff_birth = np.min(np.diff(np.sort(b)))
min_diff_death = np.min(np.diff(np.sort(d)))
min_diff = min(min_diff_birth, min_diff_death)
birth_start = np.min(b)
death_end = np.max(np.where(d != np.inf, d, 0))
grid_pts = np.linspace(birth_start, death_end, n_grid_pts)
grid_pts = np.unique(np.append(grid_pts, np.array(list(set(b) | set(d[d != np.inf])))))
offset = np.max(np.diff(grid_pts))
grid_pts = np.array([np.min(grid_pts) - offset] + list(grid_pts) + [np.max(grid_pts) + offset])

# %% 
from itertools import product
Grid = np.array([(i,j) for (i,j) in product(grid_pts, grid_pts) if i <= j])

# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=spectral_rank, i=Grid[:,0], j=Grid[:,1], p=0, w=0.0, terms=False)
# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=np.sum, i=Grid[:,0], j=Grid[:,1], p=0, w=0.0, terms=False)
# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=lambda x: np.sum(x/(x + 1e-2)), i=Grid[:,0], j=Grid[:,1], p=0, w=10.0, terms=False)
bq = BettiQuery(S, f=sublevel_filter, p=0)
bq.sign_width = 0.0
# betti_grid2 = bq(Grid[:,0], Grid[:,1], terms=False, mf = lambda x: np.sum(x**2))
# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=lambda x: np.sum(x**2), i=Grid[:,0], j=Grid[:,1], p=0, w=10.0, terms=False)
# ri = betti_grid

# %% Compute all the eigenvalues to test different matrix functions 
from pbsig.utility import BlockReduce
T1, T2, T3, T4 = BlockReduce(), BlockReduce(), BlockReduce(), BlockReduce()

for t1, t2, t3, t4 in bq.generate(i=Grid[:,0], j=Grid[:,1], mf = lambda x: x):
  T1 += t1 
  T2 += t2 
  T3 += t3
  T4 += t4
# list(bq.generate(i=Grid[0,0], j=Grid[0,1], mf = spectral_rank))
# bq(*Grid.T, spectral_rank)

sr = spectral_rank(method=0, vector_valued=True)
ri = T1.reduce(sr) - T2.reduce(transform=sr) - T3.reduce(transform=sr) + T4.reduce(transform=sr)

# %% Compute the PBN at the grid points
# eps = np.geomspace(1e-4, 1e-2, 32)
# mean_logspaced = lambda x: np.sum((np.tile(x[:,np.newaxis], 32) * np.reciprocal(x[:,np.newaxis] + eps)).mean(axis=1))
# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=mean_logspaced, i=Grid[:,0], j=Grid[:,1], p=0, w=0.05, terms=False, isometric=False, form="lo")
# betti_grid = progressbar(betti_grid, count=len(Grid))
# ri = np.array(list(betti_grid))
# assert not(np.any(np.isnan(ri)))
# bdvrd = lambda x: 0.0 if np.isclose(np.sum(x), 0) else np.sum(np.repeat(1e-6, len(x)))/np.sum(np.sqrt((np.abs(x) * (np.abs(x) + 1e-6))))
# betti_grid = betti_query(S, f=sublevel_filter, matrix_func=bdvrd, i=Grid[:,0], j=Grid[:,1], p=0, w=0.10, terms=False)

# np.sum(np.abs(ew1 - (ew1 + 1e-6))/np.sqrt((np.abs(ew1)**2 + np.abs(ew1 + 1e-6)**2)))
# np.sum(np.abs(ew2 - (ew2 + 1e-6))/np.sqrt((np.abs(ew2)**2 + np.abs(ew2 + 1e-6)**2)))

# %% 
from pbsig.color import bin_color
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap

def figure_sf(x, y, ri, **kwargs):
  from pbsig.color import color_mapper
  from pbsig.vis import valid_parameters
  cmap = kwargs.pop('cmap', color_mapper(lb=np.min(ri), ub=np.max(ri)))
  
  min_ri = np.min(ri)
  LR_map = {}
  rects, rect_ind = [], []
  for i,j,ij_val in zip(x, y, ri):
    i_ind, j_ind = np.searchsorted(grid_pts,[i,j])
    if j_ind < i_ind: assert False
    if i_ind != 0 and j_ind != len(grid_pts):
      br_x, br_y = grid_pts[i_ind], grid_pts[j_ind]
      ul_x, ul_y = grid_pts[i_ind-1], grid_pts[j_ind+1] if j_ind < (len(grid_pts) - 1) else grid_pts[j_ind] + 5*min_diff
      rects.append([np.mean([ul_x, br_x]), np.mean([ul_y, br_y]), np.abs(br_x - ul_x), np.abs(br_y - ul_y), ij_val])
      rect_ind.append([i_ind-1, i_ind, j_ind, j_ind+1])
      LR_map[br_x] = max(ij_val, LR_map.get(br_x, min_ri))
  rects = np.array(rects)
  rec_ind = np.array(rect_ind)

  # col_map = linear_cmap('value', palette = "Viridis256", low=np.min(ri), high=np.max(ri))
  rect_src = ColumnDataSource(dict(
    x=rects[:,0], y=rects[:,1], 
    width=rects[:,2], height=rects[:,3], 
    value=rects[:,4], 
    color=cmap(rects[:,4])
  ))
  tooltips = [ ("(x, y)", "($x, $y)"), ("value", "@value") ]
  fig_kwargs = dict(width=250, height=250, tools="pan,wheel_zoom,reset,hover,save", tooltips=tooltips) 
  fig_kwargs |= kwargs
  p = figure(**fig_kwargs)
  p.rect(
    x='x', y='y', width='width', height='height',
    fill_color='color', line_alpha=1.0, 
    line_color="color",
    source=rect_src
    # line_color="white"
  )
  p.outline_line_color = None

  ## Add interpolated diagonal triangles
  patches_x, patches_y = [], []
  patch_colors = []
  for i, ii in pairwise(grid_pts):
    if i in LR_map or ii in LR_map:
      tri_color = cmap(max(LR_map.get(i, 0.0), LR_map.get(ii, 0.0)))
      patches_x.append([i,i,ii])
      patches_y.append([i,ii,ii])
      patch_colors.append(tri_color)
  p.patches(xs=patches_x, ys=patches_y, fill_color=patch_colors, line_width=0.0)
  # print(tri_color)
  
  return p 

# %% 
from pbsig.linalg import tikhonov, spectral_rank, heat
tk = tikhonov(eps=1.0)
sr = spectral_rank(method=0, vector_valued=True)
spectral_ri = lambda f: T1.reduce(f) - T2.reduce(f) - T3.reduce(f) + T4.reduce(f)
 

show(figure_sf(Grid[:,0], Grid[:,1], spectral_ri(sr)))
show(figure_sf(Grid[:,0], Grid[:,1], spectral_ri(tikhonov(eps=0.01)))) ## tikhonov looks okay at w = 0.0!

from pbsig.vis import figure_plain
from bokeh.layouts import row
from bokeh.models import Div
gx, gy = Grid[:,0], Grid[:,1]

## Tikhonov row 
figs_tik = [figure_sf(gx, gy, spectral_ri(tikhonov(eps=eps)), width=120, height=120) for eps in np.geomspace(1e-5, 0.025, 5)]
for i in range(len(figs_tik)):
  figs_tik[i] = figure_dgm(figure=figs_tik[i])
  figs_tik[i] = figure_plain(figs_tik[i])
show(row(figs_tik))

## Heat trace row
figs_ht = [figure_sf(gx, gy, spectral_ri(heat(t=t, complement=True)), width=120, height=120) for t in np.flip(np.geomspace(5, 10000, 5))]
for i in range(len(figs_ht)):
  figs_ht[i] = figure_dgm(figure=figs_ht[i])
  figs_ht[i] = figure_plain(figs_ht[i])
show(row(figs_ht))

## Smoothing the function a little
# np.random.seed(1247)
# smooth_eps = (1.0 - f.smooth)*3 # 0.00075
# f2 = random_function(n_extrema=12, n_pts=1500, walk_distance=0.005, plot=True, eps=smooth_eps)
# sublevel_filter2 = sx.lower_star_filter(f2(np.linspace(0,1,150)))
# bq2 = BettiQuery(S, f=sublevel_filter2, p=0)
# bq2.sign_width = 0.0
# T1_2, T2_2, T3_2, T4_2 = BlockReduce(), BlockReduce(), BlockReduce(), BlockReduce()
# for t1, t2, t3, t4 in bq2.generate(i=Grid[:,0], j=Grid[:,1], mf = lambda x: x):
#   T1_2 += t1 
#   T2_2 += t2 
#   T3_2 += t3
#   T4_2 += t4
# spectral_ri2 = lambda f: T1_2.reduce(f) - T2_2.reduce(f) - T3_2.reduce(f) + T4_2.reduce(f)

# figs_tik2 = []
# for eps in np.geomspace(1e-5, 0.025, 5):
#   ri = spectral_ri(tikhonov(eps=eps))
#   cmap = color_mapper('viridis', np.min(ri), np.max(ri))
#   figs_tik2.append(figure_sf(gx, gy, spectral_ri2(tikhonov(eps=eps)), cmap=cmap, width=120, height=120))
# # figs_tik2 = [figure_sf(gx, gy, spectral_ri2(tikhonov(eps=eps)), width=120, height=120) for eps in np.geomspace(1e-5, 0.025, 6)]
# for i in range(len(figs_tik2)):
#   figs_tik2[i] = figure_dgm(figure=figs_tik2[i])
#   figs_tik2[i] = figure_plain(figs_tik2[i])
# show(row(figs_tik2))

## Assemble the figures 

# show(column(row(figs_tik2), row(figs_tik), row(figs_ht)))
f_p.title = "Real-valued function"
pp.title = "Rank function"
f_p.yaxis.axis_label = None # "f(x)"
# f_p.xaxis.axis_label = "Domain"
f_p.y_range = pp.y_range
f_p.xaxis.axis_label. visible = False
f_p.toolbar_location = None 
f_p.width = int(pp.width * 0.85)
f_p.height = pp.height
interp_figs = column(row(figs_tik), row(figs_ht))
for f in figs_tik: f.width = f.height = 135
for f in figs_ht: f.width = f.height = 135
div_title = Div(text="Spectral rank relaxations", styles={'font-weight':'bold'})
final_fig = row(column(f_p), column(pp), column(div_title, row(figs_tik), row(figs_ht)))
pp.xaxis.visible = False
pp.yaxis.visible = False

show(final_fig)


from bokeh.plotting import gridplot
from bokeh.io import export_svg, export_png
for f in [f_p, pp, *figs_tik, *figs_ht]:
  f.output_backend = 'svg'
export_svg(final_fig, filename="spectral_relax.svg")
# export_png(final_fig, filename="spectral_relax.png", width=1200*15, height=300*15)

# %% Exploring what makes the vertical lines constant 
# ii = np.argmin(np.abs(grid_pts - 6))
# grid_pts
# betti_query(S, f=sublevel_filter, matrix_func=mean_logspaced, i=Grid[:,0], j=Grid[:,1], p=0, w=1.0, terms=False)

# list(betti_query(S, f=sublevel_filter, matrix_func=mean_logspaced, i=grid_pts[ii], j=grid_pts[ii+3], p = 0, w=1.0))

# from pbsig.betti import betti_query
# b_nums_across_gap = list(betti_query(S, f=sublevel_filter, matrix_func=np.sum, i=np.linspace(3.477, 3.584, 132), j=np.repeat(6.038, 132), p = 0, w=1.0))
# p = figure(width=250, height=200)
# p.line(np.linspace(3.477, 3.584, 132), b_nums_across_gap)
# show(p)

def cart2bary(point: np.ndarray, vertices: np.ndarray):
  v0, v1, v2 = np.array(vertices)
  detT = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
  w0 = ((v1[1] - v2[1]) * (point[0] - v2[0]) + (v2[0] - v1[0]) * (point[1] - v2[1])) / detT
  w1 = ((v2[1] - v0[1]) * (point[0] - v2[0]) + (v0[0] - v2[0]) * (point[1] - v2[1])) / detT
  w2 = 1 - w0 - w1
  return np.array([w0, w1, w2])

from pbsig.color import color_mapper
col_map = color_mapper("viridis", lb=np.min(ri), ub=np.max(ri))

from scipy.spatial.distance import cdist
def rect_interp(R: tuple, R_coords: np.ndarray, n: int = 5):
  br, bl, tr, tl = R
  bl_tri = np.array([[0,1], [1,0], [0,0]])
  tr_tri = np.array([[0,1], [1,1], [1,0]])
  ep_x, ep_y = np.meshgrid(1.0-np.linspace(0,1,5, endpoint=False), 1.0-np.linspace(0,1,5,endpoint=False))
  eval_coords = np.c_[np.ravel(ep_x), np.ravel(ep_y)]
  which_tri = np.argmin(cdist(eval_coords, np.array([[0,0], [1,1]])), axis=1)
  bary_coords = np.array([cart2bary([x,y], bl_tri) if ti == 0 else cart2bary([x,y], tr_tri) for (x,y), ti in zip(eval_coords, which_tri)])
  assert np.allclose(bary_coords.sum(axis=1), 1.0)
  b1,b2,b3 = bary_coords.T
  b_values = np.where(which_tri == 0, b1*tl + b2*br + b3*bl, b1*tl + b2*br + b3*br)
  br_c, bl_c, tr_c, tl_c = R_coords
  nb = len(bary_coords)
  B1 = b1[:,np.newaxis]*tl_c + b2[:,np.newaxis]*br_c + b3[:,np.newaxis]*bl_c
  B2 = b1[:,np.newaxis]*tl_c + b2[:,np.newaxis]*tr_c + b3[:,np.newaxis]*br_c
  B_coords = np.where(which_tri[:,np.newaxis] == 0, B1, B2)
  return b_values, B_coords

## Make a linearly-varying rect 
N_rect = 5

min_offset = 5*np.min(np.diff(grid_pts))
# rect_value_map = dict(zip(rec_ind[:,1:3]))
for ii, (a,b,c,d) in enumerate(rect_ind):
  default_value = rect_value_map[(b,c)]
  br = rect_value_map.get((b,c), default_value)
  bl = rect_value_map.get((a,c), default_value)
  tr = rect_value_map.get((b,d), default_value)
  tl = rect_value_map.get((a,d), default_value)
  br_c = grid_pts[b], grid_pts[c]
  bl_c = grid_pts[a], grid_pts[c]
  tr_c = grid_pts[b], grid_pts[d]
  tl_c = grid_pts[a], grid_pts[d]

  R_coords = np.vstack([[br_c, bl_c, tr_c, tl_c]])
  rect_interp((br,bl,tr,tl), R_coords)
  # br_coords = 


  col_map(rect_interp((br,bl,tr,tl), 5))
  col_map([br,bl], 'hex')

  from scipy.spatial.distance import cdist
  vertices = np.array([[0,1],[0,0],[1,1],[0,1]])
  # xg, yg = 1.0-np.linspace(1/5,1,5, endpoint=True), np.linspace(0,1,5, endpoint=True)
  vertex_dists = np.array([np.ravel(cdist(np.atleast_2d([x,y]), vertices)) for x,y in product(xg,xg)])
  np.sqrt(2) - vertex_dists



  # def rect_rgba_lerp(values: tuple, n_rect: int = 5):
  #   assert len(values) == n_rect**2
  #   img = np.empty((n_rect,n_rect), dtype=np.uint32)
  #   view = img.view(dtype=np.uint8).reshape((n_rect, n_rect, 4))
  #   for i, j in product(range(n_rect), 2):
  #     view[i, j, 0] = int(i/N*255)
  #     view[i, j, 1] = 158
  #     view[i, j, 2] = int(j/N*255)
  #     view[i, j, 3] = 255


## TODO: get rectangle at a point, in the grid, draw a rect with a fixed color
## can add a way to interpolate later via e.g. image texture or something
np.unique(ri)

# K = filtration(S, f=sublevel_filter)
# B1 = boundary_matrix(K, p = 1).todense()
# p_ind = sublevel_filter(faces(K,0)) > 2
# q_ind = sublevel_filter(faces(K,1)) <= 2
# np.linalg.matrix_rank(B1[np.ix_(p_ind, q_ind)])

# np.linalg.matrix_rank(B1.todense()[:,])


# %% 
S = delaunay_complex(X)
list(zip(f_ecc(S), map(tuple, S)))

f = sx.fixed_filter(S, f_ecc(S))
K = sx.filtrations.RankFiltration(S, f = f)
dgm = ph(K, collapse=False)

f_index = sx.fixed_filter(list(faces(K)), np.arange(len(K)))
K_index = sx.filtrations.RankFiltration(S, f = f_index)
dgm = ph(K_index, collapse=False)

show(figure_dgm(dgm[1]))


