import numpy as np 
from typing import * 
import pyflubber
from svgpathtools import svg2paths  
from svgpathtools import parse_path, Line, Path, wsvg
import bokeh
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()
from pbsig.vis import figure_point_cloud

## From: https://github.com/veltman/flubber/blob/master/demos/medley.html
# svg2paths("/Users/mpiekenbrock/pbsig/src/pbsig/data/flubber_svgs/star.svg")
star_path = parse_path("M480,50L423.8,182.6L280,194.8L389.2,289.4L356.4,430L480,355.4L480,355.4L603.6,430L570.8,289.4L680,194.8L536.2,182.6Z")
heart_path = parse_path("M480,437l-29-26.4c-103-93.4-171-155-171-230.6c0-61.6,48.4-110,110-110c34.8,0,68.2,16.2,90,41.8 C501.8,86.2,535.2,70,570,70c61.6,0,110,48.4,110,110c0,75.6-68,137.2-171,230.8L480,437z")
horse_path = parse_path("M595,82.1c1,1-1,2-1,2s-6.9,2-8.9,4.9c-2,2-4.9,8.8-4.9,8.8c3.9-1,8.9-2,13.8-4c1,0,2,1,3,2c1,0-11.8,4.9-14.8,6.9c-2,2-11.8,9.9-14.8,9.9c-2.9,0-9.9,1-9.9,1c1,2,2,3.9,3.9,6.9c0,0-6.9,4-6.9,4.9c-1,1-5.9,6.9-5.9,6.9s17.7,1.9,23.6-7.9c-5.9,9.8-19.7,19.7-48.2,19.7c-29.5,0-53.1-11.8-68.9-17.7c-16.7-6.9-38.4-14.8-56.1-14.8c-16.7,0-36.4,4.9-49.2,16.8c-22.6-8.8-54.1-4-68.9,9.8c-13.8,13.8-27.5,30.5-29.5,42.3c-2.9,12.9-9.8,43.3-19.7,57.2c-13.8,22.5-29.5,28.5-34.5,38.3c-4.9,9.9-4.9,30.5-4,30.5c2,1,8.9,0,12.8-2c7.9-2.9,29.5-25.6,37.4-36.4c7.9-10.9,34.5-58.1,38.4-74.8s7.9-33.5,19.7-42.3c12.8-8.8,28.5-4.9,28.5-3.9c0,0-14.7,11.8-15.7,44.3s18.7,28.6,8.8,49.2c-9.9,17.7-39.3,5.9-49.2,16.7c-7.9,8.9,0,40.3,0,46.2c0,6-3,33.5-4.9,40.4c-1,5.9,0,9.8-1,13.8c-1,3,6,3.9,6,3.9s-6,7.8-8.9,5.9c-2.9-1-4.9-1-6.9,0c-2,0-5.9,1.9-9.9,0L232.9,401c2,1,4.9,1.9,7.9,1c4-1,23.6-9.9,25.6-11.9c2.9-1,19.7-10.8,22.6-16.7c2-5.9,5.9-24.6,5.9-30.5c1-6,2-24.6,2-29.5s-1-13.8,0-17.7c2-2.9,4.9-6.9,8.9-8.9c4.9-1,10.8-1,11.8-1c2,0,18.7,2,21.6,2c3.9,0,19.7-2.9,23.6-5c4.9-0.9,7.8,0,8.9,2c2,1.9-2,4.9-2,5.9c-1,1-8.8,10.8-10.8,14.7c-2,4.9-8.8,13.8-6.9,17.7c2,3.9,2,4.9,7.8,7.9c5.9,1.9,28.5,13.8,41.3,25.6c13.8,12.7,26.6,28.4,28.6,36.4c2.9,8.9,7.8,9.8,10.8,9.8c3,1,8.9,2,8.9,5.9s-1,8.8-1,8.8l36.4,13.8c0,0,0-12.8-1-17.7c-1-5.9-6.9-11.8-11.8-17.7c-4.9-6.9-56-57.1-61-61c-4.9-3-8.9-6.9-9.8-14.7c-1-7.9,8.8-13.8,14.8-20.6c3.9-4.9,14.7-27.6,16.7-30.6c2-2.9,8.9-10.8,12.8-10.8c4.9,0,15.8,6.9,29.5,11.8c5.9,2,48.2,12.8,54.1,14.8c5.9,1,18.6,0,22.6,3.9c3.9,2.9,0,10.9-1,15.8c-1,5.9-11.8,27.5-11.8,27.5s2,7.8,2,13.8c0,6.9-2.9,31.5-5.9,39.3c-2,8.9-15.8,31.6-18.7,35.5c-2,2.9-4.9,4.9-4.9,9.9c0,4.9,8.8,6,11.8,9.8c4,3,1,8.8,0,14.8l39.4,16.7c0-2.9,2-7.9,0-9.9c-1-2.9-5.9-8.8-8.8-12.8c-2-2.9-8.9-13.8-10.8-15.8c-2-2.9-2-8.8,0-13.8c1-4.9,13.8-38.3,14.7-42.3c2-4.9,20.7-44.3,22.6-49.2c2-5.9,17.7-34.4,19.7-39.4c2-5.9,14.8-10.8,18.7-10.8c4.9,0,29.5,8.8,33.4,10.8c2.9,1,25.6,10.9,29.5,12.8c4.9,1.9,2,5.9-1,6.9c-2.9,1.9-39.4,26.5-42.3,27.5c-2.9,1-5.9,3.9-7.9,3.9c-2.9,0-6.9,3.1-6.9,4c0,2-1,5.9-5.9,5.9c-3.9,0-11.8-5.9-16.7-11.8c-6.9,3.9-11.8,6.9-14.8,12.8c-4.9,4.9-6.9,8.9-9.8,15.8c2,2,5.9,2.9,8.8,2.9h31.5c3,0,6.9-0.9,9.9-1.9c2.9-2,80.7-53.1,80.7-53.1s12.8-9.9,12.8-18.7c0-6.9-5.9-8.9-7.9-11.8c-3-1.9-20.7-13.8-23.6-15.7c-4-2.9-17.7-10.9-21.6-12.9c-3-1.9-13.8-5.8-13.8-5.8c3-8.9,5-15.8,5.9-17.7c1-2,1-19.7,2-22.7c0-2.9,5-15.7,6.9-17.7c2-2,6.9-17.7,7.9-20.7c1-1.9,8.8-24.6,12.8-24.6c3.9-1,7.9,2.9,11.8,2.9c4,1,18.7-1,26.6,0c6.9,1,15.8,9.9,17.7,10.8c2.9,1,9.8,3.9,11.8,3.9c1,0,10.8-6.9,10.8-8.8c0-2-6.9-5.9-7.9-5.9c-1-1-7.8-4.9-7.8-4.9c0,1,2.9-1.9,7.8-1.9c3.9,0,7.9,3.9,8.8,4.9c2,1,6.9,3.9,7.9,1.9c1-1,4.9-5.9,4.9-8.9c0-4-3.9-8.8-5.9-10.8s-24.6-23.6-26.6-24.6c-2.9-1-14.7-11.8-14.7-14.7c-1-2-6.9-6.9-7.9-7.9s-30.5-21.6-34.5-24.6c-3.9-2.9-7.9-7.8-7.9-12.7s-2-17.7-2-17.7s-6.9-1-9.8,1.9c-2.9,2-9.8,17.8-13.8,17.8c-10.9-2-24.6,1-24.6,2.9c1,2.9,10.8,1,10.8,1c0,1-3.9,5.9-6.9,5.9c-2,0-7.8,2-8.8,2.9c-2,0-5.9,3.1-5.9,3.1c2.9,0,5.9,0,9.8,0.9c0,0-5.9,4-8.9,4c-2.9,0-12.8,2.9-15.7,3.9c-2,1.9-9.9,7.9-9.9,7.9H589l1,2h4.9L595,82.1L595,82.1z")
chair_path = parse_path("M638.9,259.3v-23.8H380.4c-0.7-103.8-37.3-200.6-37.3-200.6s-8.5,0-22.1,0C369.7,223,341.4,465,341.4,465h22.1 c0,0,11.4-89.5,15.8-191h210.2l11.9,191h22.1c0,0-5.3-96.6-0.6-205.7H638.9z")
hand_path = parse_path("M23 5.5V20c0 2.2-1.8 4-4 4h-7.3c-1.08 0-2.1-.43-2.85-1.19L1 14.83s1.26-1.23 1.3-1.25c.22-.19.49-.29.79-.29.22 0 .42.06.6.16.04.01 4.31 2.46 4.31 2.46V4c0-.83.67-1.5 1.5-1.5S11 3.17 11 4v7h1V1.5c0-.83.67-1.5 1.5-1.5S15 .67 15 1.5V11h1V2.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5V11h1V5.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5z")
plane_path = parse_path("M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z")
bolt_path = parse_path("M7 2v11h3v9l7-12h-4l4-8z")
note_path = parse_path("M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z")
#   <path id="triforce" d="M345.47,250L460.94,450L230,450Z M460.94,50L576.41,250L345.47,250Z M576.41,250L691.88,450L460.94,450Z"/>
shape_paths = [star_path, heart_path, horse_path, chair_path, hand_path, plane_path, bolt_path, note_path]

from pbsig.shape import PL_path
from pbsig.pht import parameterize_dt, stratify_sphere, normalize_shape
DV = stratify_sphere(d=1, n=64)                          ## direction vectors

from pbsig.itertools import LazyIterable
load_shape = lambda index: normalize_shape(PL_path(shape_paths[index], 100, close_path=True, output_path=False), DV)
SHAPE_PCS = LazyIterable(load_shape, count=len(shape_paths))

## Build a continuous family of interpolators between shapes
import pyflubber
from more_itertools import pairwise
from pbsig.interpolate import LinearFuncInterp
# f_interp = [pyflubber.interpolator(s1, s2, closed=True) for s1, s2 in pairwise(SHAPE_PCS)]
# F_interp = LinearFuncInterp(f_interp)
F_interp = pyflubber.interpolator(SHAPE_PCS[0], SHAPE_PCS[1], closed=True) 
SHAPE_INTERP = LazyIterable(lambda index: F_interp(float(index/100)), count=100)

## Generate the parameterized family of filter functions
from pbsig.itertools import rproduct, rstarmap
from splex import lower_star_weight
parameter_space = rproduct(SHAPE_INTERP, DV) 
filter_mesh = lambda mesh, v: lower_star_weight(mesh @ v)
filter_family = rstarmap(filter_mesh, parameter_space)

## Build the sieve 
from pbsig.linalg import eigvalsh_solver
from pbsig.betti import Sieve
from splex import *
n = len(SHAPE_INTERP[0])
S = simplicial_complex(pairwise(range(n)))
sieve = Sieve(S, family=filter_family, p=0, form='lo')

## Choose a rectilinear pattern 
sieve.randomize_pattern(6)
show(sieve.figure_pattern())

## Set solve parameters and sift away
# import line_profiler
# profile = line_profiler.LineProfiler()
# profile.add_function(sieve.sift)
# profile.add_function(sieve.project)
# profile.add_function(sieve.solver.__call__)
# profile.add_function(sieve.solver.solver.__call__)
# profile.add_function(product)
# profile.enable_by_count()
sieve.solver = eigvalsh_solver(sieve.laplacian, tolerance=1e-5)
sieve.sift(w=0.15, progress=True, k=15)
# profile.print_stats(output_unit=1e-3, stripzeros=True)



## Plot the signals
from bokeh.io import export_png
from bokeh.models import Div 
from scipy.signal import savgol_filter
from bokeh.models import Range1d
from pbsig.vis import figure_dgm 
summary = sieve.summarize()
n_dir = 64
n_summary, sum_len = summary.shape
pic_dir = "/Users/mpiekenbrock/pbsig/animations/flubber_svgs/star2heart/smooth"

def figure_signal(x: ArrayLike, y: ArrayLike, figkwargs = dict(), **kwargs):
  p = figure(**figkwargs)
  p.line(x, y, **kwargs)
  return p

BOUNDS = {}
for j in range(n_summary):
  min_bnds = savgol_filter(np.array([np.array(s).min() for s in chunked(summary[j], n_dir)]), 15, 3) 
  max_bnds = savgol_filter(np.array([np.array(s).max() for s in chunked(summary[j], n_dir)]), 15, 3) 
  min_bnds = np.minimum(min_bnds, -1)
  max_bnds = np.maximum(max_bnds, 1)
  BOUNDS[j] = {'min' : min_bnds, 'max': max_bnds}

summary_colors = getattr(bokeh.palettes, f"Viridis{n_summary}")
for i in range(sum_len // n_dir):
  
  ## Obtain the shape plots
  shape_fig = figure(height=80*3, width=80*3)
  shape_fig.toolbar_location = None 
  X = np.c_[SHAPE_INTERP[i][:,0], -SHAPE_INTERP[i][:,1]]
  cycle_outline = np.vstack([X, X[0]])
  shape_fig.patch(*cycle_outline.T, fill_alpha = 0.35)
  # p.line(*cycle_outline.T, line_color='red')
  # p.scatter(*SHAPE_INTERP[i].T)
  shape_fig.xaxis.visible = False
  shape_fig.yaxis.visible = False

  ## Obtain the signature plots  
  # p = figure()
  # p.scatter(np.arange(100), max_bnds)
  # show(p)
  figs = []
  for j, col in zip(range(n_summary), summary_colors):
    # bnds = summary[j].min(), summary[j].max()
    bnds = BOUNDS[j]['min'][i], BOUNDS[j]['max'][i] 
    extra = abs(float(np.diff(bnds)))*0.15
    x = np.arange(n_dir)
    y = savgol_filter(summary[j][i*n_dir:((i+1)*n_dir)], 8, 3)
    # bnds = y.min()-0.5, y.max()+0.5
    bnds = bnds[0]-extra, bnds[1]+extra
    sig = figure_signal(x, y, figkwargs=dict(width=250, height=80), line_color=col, line_width=2)
    sig.toolbar_location = None
    sig.y_range = Range1d(*bnds)
    sig.xaxis.visible = False
    sig.yaxis.minor_tick_line_color = None  
    figs.append(sig)

  ## Construct the diagram plot
  dgm_plot = figure_dgm(width=240, height=240, title=None)
  indices = sieve._pattern['index']
  for idx, col in zip(np.unique(indices), summary_colors):
    r = sieve._pattern[indices == idx]
    x,y = np.mean(r['i']), np.mean(r['j'])
    w,h = np.max(np.diff(np.sort(r['i']))), np.max(np.diff(np.sort(r['j'])))
    dgm_plot.rect(x,y,width=w,height=h,fill_color=col, line_color=None)
  dgm_plot.toolbar_location = None
  dgm_plot.xaxis.visible = False 
  dgm_plot.yaxis.visible = False 
  
  ## Aggregate the three plots
  # plot_titles = row(
  #   Div(text="<strong> PL-embedded complex </strong>", styles={"background-color": "white"}, width=240), 
  #   Div(text="<strong> Spectral rank signatures </strong>",  styles={"background-color": "white"}, width=500), 
  #   Div(text="<strong> Random sieve </strong>", styles={"background-color": "white"}, width=240)
  # )
  full_plot = column(
    row(shape_fig, column(*figs[:3]), column(*figs[3:]), dgm_plot)
  )
  #show(full_plot)
  export_png(full_plot, filename=pic_dir+f"/frame_{i}.png")

# ----------------------
import time
from splex import * 
from pbsig.pht import stratify_sphere
from pbsig.betti import Sieve

## Plot the image contours
# X = SHAPE_INTERP[100]
X = SHAPE_PCS[3]
f = lower_star_weight(X @ np.array([1,0]))
sieve_image = Sieve(S, family=[f],  p=0, form='lo')
# sieve_image.solver = 

from pbsig.linalg import sgn_approx
phi = sgn_approx(eps=0.001, p=1.0)

xi, yi = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
z = np.zeros(xi.shape)
for v in stratify_sphere(1, 32):
  f = lower_star_weight(X @ v)
  z += np.reshape([np.sum(phi(sieve_image.project(i,j,w=0.25,f=f, k=10))) if i < j else 0.0 for i,j in zip(np.ravel(xi),np.ravel(yi))], xi.shape)
  print(v)

from bokeh.palettes import Sunset10
p = figure(width=300, height=300)
levels = np.unique(np.quantile(np.ravel(z[z!=0]), q=np.linspace(0.0,1.0,10)))
p.contour(xi, yi, z, levels=levels, fill_color=Sunset10, line_color="black")
show(p)
#time.sleep(0.25)

from pbsig.persistence import ph
from pbsig.vis import figure_dgm
X = SHAPE_PCS[2]
zero_dgms = []
K = filtration(S, f=lambda s: s)
for i, v in enumerate(stratify_sphere(d=1, n=100)):
  K.reindex(lower_star_weight(X @ v))
  dgm0 = ph(K, engine="dionysus")[0]
  inessential = dgm0['death'] != np.inf
  dgm0_ext = np.c_[dgm0['birth'][inessential], dgm0['death'][inessential], np.repeat(i, sum(inessential))]
  zero_dgms.append(dgm0_ext)
zero_dgms = np.vstack(zero_dgms)

from pbsig.color import bin_color
from bokeh.models import Range1d
color_pal = bokeh.palettes.viridis(8)
colors = (bin_color(zero_dgms[:,2].astype(int), 'viridis')*255).astype(np.uint8)

p = figure_dgm()
p.scatter(zero_dgms[:,0], zero_dgms[:,1], color=colors)
p.x_range = Range1d(zero_dgms[:,0].min()-0.1, zero_dgms[:,1].max()+0.1)
p.y_range = Range1d(zero_dgms[:,0].min()-0.1, zero_dgms[:,1].max()+0.1)
show(p)


uhp_pts = rproduct(np.linspace(0,1,50), np.linspace(0,1,50))
uhp_pts = np.array([(i,j) for i,j in image_family if i < j])

# sieve_image._pattern = np.fromiter(
#   zip(uhp_pts[:,0], uhp_pts[:,1], np.repeat(1, len(uhp_pts)), np.arange(len(uhp_pts))),
#   dtype=sieve_image._pattern.dtype
# )
# for i,j in zip(np.ravel(xi),np.ravel(yi)):
#   if i < j:
#     sum(sieve_image.project(i,j,w=0.15, f=f))

# show(sieve_image.figure_pattern())

sieve_image.solver = eigvalsh_solver(sieve_image.laplacian, tolerance=1e-5)
sieve_image.sift(w=0.15, progress=True, k=15)

summary_image = sieve_image.summarize()

from bokeh.palettes import Sunset8
from pbsig.color import bin_color
v_cols = bokeh.palettes.viridis(10)


p.image_rgba()









## Vineyard diagrams?
ph(filtration(S, filter_family[0]))

from scipy.sparse.linalg import eigsh
eigsh(sieve.laplacian)


import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(perfect_hash_dag)
profile.enable_by_count()
profile.print_stats(output_unit=1e-3, stripzeros=True)




## Indeed, the procrustes differences are all low
diffs = np.array([procrustes_dist_cc(F_interp(t0), F_interp(t1)) for t0, t1 in pairwise(np.linspace(0, 1, 100))])

## Show the shapes
Xt = [F_interp(t) for t in np.linspace(0.0, 1.0, 35)]
p = row([figure_point_cloud(S, width=200, height=200) for S in Xt])
show(p)

## Construct the random sieve
# from pbsig.utility import RepeatableIterable
# RepeatableIterable()


# def flubber_filter(F: Callable, time_points: ArrayLike) -> Iterable[Callable]: 
  
#   class _family:
    
#   for t in time_points: 
#     X = F(t)
#     DT = parameterize_dt(X, dv=DV, normalize=False)
#     for f in DT:
#       yield f

from more_itertools import seekable

family = flubber_filter(F_interp, np.linspace(0,1,100))



## Show the shapes
# p = column([
#   row([figure_point_cloud(S) for S in shapes_emb[:4]]),
#   row([figure_point_cloud(S) for S in shapes_emb[4:]])
# ])
# show(p)



# u = shape_center(P, method="directions", V=V_dir)
# P = P - u 
# L = sum([-np.min(P @ vi[:,np.newaxis]) for vi in V_dir])
# P = (1/L)*P





# s_star = np.array([complex2pt(pt.start) for pt in star_path] + [complex2pt(star_path[0].start)])
s_heart = np.array([complex2pt(pt) for pt in heart_pts])

p = figure()
p.scatter(s_heart[:,0], s_heart[:,1])
show(p)


f_interp = pyflubber.interpolator(s_star, s_heart, closed=True)




p = figure(width=300, height=300)
p.line(*f_interp(0.9).T)
show(p)

f_interp(1.)



  # <path id="star" d="M480,50L423.8,182.6L280,194.8L389.2,289.4L356.4,430L480,355.4L480,355.4L603.6,430L570.8,289.4L680,194.8L536.2,182.6Z"/>
  # <path id="heart" d="M480,437l-29-26.4c-103-93.4-171-155-171-230.6c0-61.6,48.4-110,110-110c34.8,0,68.2,16.2,90,41.8
  # 	C501.8,86.2,535.2,70,570,70c61.6,0,110,48.4,110,110c0,75.6-68,137.2-171,230.8L480,437z"/>
  # <path id="horse" d="M595,82.1c1,1-1,2-1,2s-6.9,2-8.9,4.9c-2,2-4.9,8.8-4.9,8.8c3.9-1,8.9-2,13.8-4c1,0,2,1,3,2c1,0-11.8,4.9-14.8,6.9c-2,2-11.8,9.9-14.8,9.9c-2.9,0-9.9,1-9.9,1c1,2,2,3.9,3.9,6.9c0,0-6.9,4-6.9,4.9c-1,1-5.9,6.9-5.9,6.9s17.7,1.9,23.6-7.9c-5.9,9.8-19.7,19.7-48.2,19.7c-29.5,0-53.1-11.8-68.9-17.7c-16.7-6.9-38.4-14.8-56.1-14.8c-16.7,0-36.4,4.9-49.2,16.8c-22.6-8.8-54.1-4-68.9,9.8c-13.8,13.8-27.5,30.5-29.5,42.3c-2.9,12.9-9.8,43.3-19.7,57.2c-13.8,22.5-29.5,28.5-34.5,38.3c-4.9,9.9-4.9,30.5-4,30.5c2,1,8.9,0,12.8-2c7.9-2.9,29.5-25.6,37.4-36.4c7.9-10.9,34.5-58.1,38.4-74.8s7.9-33.5,19.7-42.3c12.8-8.8,28.5-4.9,28.5-3.9c0,0-14.7,11.8-15.7,44.3s18.7,28.6,8.8,49.2c-9.9,17.7-39.3,5.9-49.2,16.7c-7.9,8.9,0,40.3,0,46.2c0,6-3,33.5-4.9,40.4c-1,5.9,0,9.8-1,13.8c-1,3,6,3.9,6,3.9s-6,7.8-8.9,5.9c-2.9-1-4.9-1-6.9,0c-2,0-5.9,1.9-9.9,0L232.9,401c2,1,4.9,1.9,7.9,1c4-1,23.6-9.9,25.6-11.9c2.9-1,19.7-10.8,22.6-16.7c2-5.9,5.9-24.6,5.9-30.5c1-6,2-24.6,2-29.5s-1-13.8,0-17.7c2-2.9,4.9-6.9,8.9-8.9c4.9-1,10.8-1,11.8-1c2,0,18.7,2,21.6,2c3.9,0,19.7-2.9,23.6-5c4.9-0.9,7.8,0,8.9,2c2,1.9-2,4.9-2,5.9c-1,1-8.8,10.8-10.8,14.7c-2,4.9-8.8,13.8-6.9,17.7c2,3.9,2,4.9,7.8,7.9c5.9,1.9,28.5,13.8,41.3,25.6c13.8,12.7,26.6,28.4,28.6,36.4c2.9,8.9,7.8,9.8,10.8,9.8c3,1,8.9,2,8.9,5.9s-1,8.8-1,8.8l36.4,13.8c0,0,0-12.8-1-17.7c-1-5.9-6.9-11.8-11.8-17.7c-4.9-6.9-56-57.1-61-61c-4.9-3-8.9-6.9-9.8-14.7c-1-7.9,8.8-13.8,14.8-20.6c3.9-4.9,14.7-27.6,16.7-30.6c2-2.9,8.9-10.8,12.8-10.8c4.9,0,15.8,6.9,29.5,11.8c5.9,2,48.2,12.8,54.1,14.8c5.9,1,18.6,0,22.6,3.9c3.9,2.9,0,10.9-1,15.8c-1,5.9-11.8,27.5-11.8,27.5s2,7.8,2,13.8c0,6.9-2.9,31.5-5.9,39.3c-2,8.9-15.8,31.6-18.7,35.5c-2,2.9-4.9,4.9-4.9,9.9c0,4.9,8.8,6,11.8,9.8c4,3,1,8.8,0,14.8l39.4,16.7c0-2.9,2-7.9,0-9.9c-1-2.9-5.9-8.8-8.8-12.8c-2-2.9-8.9-13.8-10.8-15.8c-2-2.9-2-8.8,0-13.8c1-4.9,13.8-38.3,14.7-42.3c2-4.9,20.7-44.3,22.6-49.2c2-5.9,17.7-34.4,19.7-39.4c2-5.9,14.8-10.8,18.7-10.8c4.9,0,29.5,8.8,33.4,10.8c2.9,1,25.6,10.9,29.5,12.8c4.9,1.9,2,5.9-1,6.9c-2.9,1.9-39.4,26.5-42.3,27.5c-2.9,1-5.9,3.9-7.9,3.9c-2.9,0-6.9,3.1-6.9,4c0,2-1,5.9-5.9,5.9c-3.9,0-11.8-5.9-16.7-11.8c-6.9,3.9-11.8,6.9-14.8,12.8c-4.9,4.9-6.9,8.9-9.8,15.8c2,2,5.9,2.9,8.8,2.9h31.5c3,0,6.9-0.9,9.9-1.9c2.9-2,80.7-53.1,80.7-53.1s12.8-9.9,12.8-18.7c0-6.9-5.9-8.9-7.9-11.8c-3-1.9-20.7-13.8-23.6-15.7c-4-2.9-17.7-10.9-21.6-12.9c-3-1.9-13.8-5.8-13.8-5.8c3-8.9,5-15.8,5.9-17.7c1-2,1-19.7,2-22.7c0-2.9,5-15.7,6.9-17.7c2-2,6.9-17.7,7.9-20.7c1-1.9,8.8-24.6,12.8-24.6c3.9-1,7.9,2.9,11.8,2.9c4,1,18.7-1,26.6,0c6.9,1,15.8,9.9,17.7,10.8c2.9,1,9.8,3.9,11.8,3.9c1,0,10.8-6.9,10.8-8.8c0-2-6.9-5.9-7.9-5.9c-1-1-7.8-4.9-7.8-4.9c0,1,2.9-1.9,7.8-1.9c3.9,0,7.9,3.9,8.8,4.9c2,1,6.9,3.9,7.9,1.9c1-1,4.9-5.9,4.9-8.9c0-4-3.9-8.8-5.9-10.8s-24.6-23.6-26.6-24.6c-2.9-1-14.7-11.8-14.7-14.7c-1-2-6.9-6.9-7.9-7.9s-30.5-21.6-34.5-24.6c-3.9-2.9-7.9-7.8-7.9-12.7s-2-17.7-2-17.7s-6.9-1-9.8,1.9c-2.9,2-9.8,17.8-13.8,17.8c-10.9-2-24.6,1-24.6,2.9c1,2.9,10.8,1,10.8,1c0,1-3.9,5.9-6.9,5.9c-2,0-7.8,2-8.8,2.9c-2,0-5.9,3.1-5.9,3.1c2.9,0,5.9,0,9.8,0.9c0,0-5.9,4-8.9,4c-2.9,0-12.8,2.9-15.7,3.9c-2,1.9-9.9,7.9-9.9,7.9H589l1,2h4.9L595,82.1L595,82.1z"/>
  # <path id="chair" d="M638.9,259.3v-23.8H380.4c-0.7-103.8-37.3-200.6-37.3-200.6s-8.5,0-22.1,0C369.7,223,341.4,465,341.4,465h22.1
  # 	c0,0,11.4-89.5,15.8-191h210.2l11.9,191h22.1c0,0-5.3-96.6-0.6-205.7H638.9z"/>
  # <path id="triforce" d="M345.47,250L460.94,450L230,450Z M460.94,50L576.41,250L345.47,250Z M576.41,250L691.88,450L460.94,450Z"/>



 
from pbsig.itertools import rproduct
from array import array
from pbsig.itertools import sequence
x, y = array('I'), array('I')
x.extend([1,2,3])
y.extend([4,5,6])

s = sequence(x)
assert id(x) == id(s._seq)
view = SequenceView(s)
id(view._target)

## Side effects work! 
x, y = [0,1,2,4], [4,5,6]
x_view = SequenceView(x)
assert id(x_view._target) == id(x)
s = sequence(x)
assert id(s._seq) == id(x)
S = SequenceView(s)
assert id(S._target._seq) == id(x)
r = rproduct(x,y)
x.extend([5]) 