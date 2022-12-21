# dashboard.py
import time
from typing import *

import numpy as np 
from numpy.typing import ArrayLike

import bokeh 
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, show, curdoc
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.events import Press, PressUp
#from bokeh.models import Range1d, ColumnDataSource, BoxEditTool, Model, Annulus, Plot, AnnularWedge, Slider, Span, Panel, Tabs, Button, Div, Line, PolyAnnotation, Rect, Step, MultiChoice
from bokeh.models import *
from bokeh.core.property.color import Color 
from bokeh.transform import linear_cmap

from pbsig.color import bin_color
from pbsig.pht import rotate_S1
from pbsig.persistence import ph0_lower_star
from pbsig.pht import rotate_S1, pht_preprocess_pc
from pbsig.betti import lower_star_multiplicity
from pbsig.utility import cycle_window
from pbsig.datasets import mpeg7
from pbsig.simplicial import delaunay_complex
nan = float('nan')
# output_notebook(verbose=False, hide_banner=True)



ALL_RECTANGLES = []

def signature_plot(name: str, width: int, height: int, n_theta: int = 1000):
  sp = figure(title="Directional transform", x_axis_label="Angle", y_axis_label=None, width=width, height=height, tools=TOOLS+['xpan']) # r"\[\beta_p(K)\]"
  sp.toolbar.logo = None
  sp.y_range = Range1d(-1.0, 1.0) # bounds=(lb, ub)
  sp.x_range = Range1d(0, 2*np.pi)

  ## For step functions
  LS, SS = ColumnDataSource({ 'x' : [], 'y' : [] }), ColumnDataSource({ 'x' : [], 'y' : [] })
  step_renderer = sp.step(x="x", y="y", line_color="#f46d43", mode="before", source=SS)
  step_renderer.tags = [name+".step"]
  line_renderer = sp.line(x="x", y="y", line_color="#f46d43", source=LS)
  line_renderer.tags = [name+".line"]
  return sp

def update_guide_lines(gr, rx, ry, rw, rh, theta):
  a,b,c,d = rx-rw/2, rx+rw/2, ry-rh/2, ry+rh/2
  XY_p = np.c_[np.repeat([a,b,c,d],2), np.tile([-1.5, 1.5], 4)]
  RM = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
  XY_p = (XY_p @ RM)
  XX = np.array_split(XY_p[:,0],4)
  YY = np.array_split(XY_p[:,1],4)
  gr.data_source.data = { 'xs' : XX, 'ys' : YY, 'line_color': bin_color([a,b,c,d], "turbo", lb=-1.0, ub=1.0)}

from bokeh.events import DoubleTap, Tap, MouseEnter, ButtonClick, SelectionGeometry, Press
def box_changed_cb(attr: str, old: List[int], new: List[int]) -> None:
  #log_msgs.append(f"boxes modified: {str(old)} -> {str(new)} selected")
  # log_msgs.append("Saved rectangles: " + str(R))
  pass
  # new
  # if gr is not None: 
  #   update_guide_lines(gr, rx, ry, rw, rh, theta)


def box_tapped_cb(event) -> None:
  print(str(event))
  log_msgs.append(f"box selected: selected")

def _box_tool(plot, name: str):
  box_data = {'x': [], 'y': [], 'width': [], 'height': [], 'alpha': []}
  box_source = ColumnDataSource(data=box_data)
  box_renderer = plot.rect('x', 'y', 'width', 'height', alpha='alpha', source=box_source, fill_color="gray", fill_alpha=0.30, line_color=None)
  box_renderer.tags = [name+'.boxes']
  selected_box = Rect(fill_color="firebrick", fill_alpha=0.90, line_color="black")
  unselected_box = Rect(fill_color="gray", fill_alpha=0.30, line_color=None)
  box_renderer.selection_glyph = selected_box
  box_renderer.nonselection_glyph = unselected_box
  box_tool = BoxEditTool(renderers=[box_renderer])
  box_source.selected.on_change('indices', box_changed_cb)
  # plot_tap = plot.select(type=TapTool)
  # plot_tap.on_event("tap", box_tapped_cb)
  return box_tool

def dgm_plot(name: str, width: int, height: int):
  ## Halflane/dgm plot: sets up halfplane on [lb,ub]
  lb, ub = 0.0, 1.0
  hp = figure(title="Persistence diagram", x_axis_label="birth", y_axis_label="death", width=w, height=h, tools=TOOLS+['tap', 'crosshair'], toolbar_location="right")
  hp.line([-100,100], [-100,100], color="black", line_width=1)
  hp.toolbar.logo = None
  hp.y_range = Range1d(lb, ub) # bounds=(lb, ub)
  hp.x_range = Range1d(lb, ub) #  bounds=(lb, ub)
  lhp = PolyAnnotation(fill_color="gray",fill_alpha=0.95,xs=[-100, 100, 100, -100],ys=[-100, -100, 100, -100])
  lhp.level = 'underlay'
  hp.add_layout(lhp)
  hp.toolbar.active_tap = None
  dgm_point_src = ColumnDataSource({ 'x': [], 'y': [], 'theta': [], 'alpha': [], 'size': [] }) 
  turbo_color = linear_cmap('theta', 'Turbo256', low=0, high=2*np.pi)
  dgm_renderer = hp.scatter(x='x', y='y', size='size', marker="dot", color=turbo_color, source=dgm_point_src)
  dgm_renderer.tags = [name+'.dgm_points']
  return(hp)

def render_complex(plot, X: ArrayLike, E: ArrayLike = [], T: ArrayLike = []):
  X_src = ColumnDataSource({ 'x': X[:,0], 'y': X[:,1] , 'lambda': X[:,0] })
  turbo_color = linear_cmap('lambda', 'Turbo256', low=-1.0, high=1.0) # DataSpec dict

  ## Triangles
  TX = [list(X[t,0]) for t in T] if len(T) > 0 else []
  TY = [list(X[t,1]) for t in T] if len(T) > 0 else []
  t_renderer = plot.patches(TX, TY, color="green", alpha=0.15, line_width=0)
  
  ## Edges 
  EX = [[X[u,0], X[v,0]] for u,v in E] if len(E) > 0 else []
  EY = [[X[u,1], X[v,1]] for u,v in E] if len(E) > 0 else []
  e_renderer = plot.multi_line(EX, EY, alpha=0.80, color="firebrick", line_width=1.5)

  ## Vertices; colored by lambda
  v_renderer = plot.circle('x', 'y', size=4, alpha=1.0, color=turbo_color, source=X_src) # TODO: learn about bokeh.transforms
  return v_renderer, e_renderer, t_renderer

def circle_plot(name: str, X: ArrayLike):
  cp = figure(title="Filtration direction", plot_width=300, plot_height=300, min_border=0, toolbar_location=None, match_aspect=True, aspect_scale=1)
  cp.frame_width = cp.frame_height = 220
  cp.x_range = cp.y_range = Range1d(start=-1.7,end=1.7,bounds=(-1.7,1.7))
  S1_renderer = cp.circle(x=0, y=0, radius=1.5, line_color="black", fill_color="#7fc97f", line_width=3, fill_alpha=0.0, radius_units='data')
  ann_source = ColumnDataSource({ 'x' : [], 'y': [], 'ir': [], 'or': [], 'color': [], 'alpha': [] })
  annuli_renderer = cp.annulus(x='x', y='y', inner_radius='ir', outer_radius='or', fill_color='color', fill_alpha='alpha', line_alpha=0.0, source=ann_source)
  annuli_renderer.tags = [name + ".annuli"]

  abcd_line_src = ColumnDataSource({ 'xs' : [], 'ys': [], 'line_color': [] })
  guideline_renderer = cp.multi_line(xs='xs', ys='ys', line_width=1.5, color='line_color', line_dash="dashed", source=abcd_line_src)
  guideline_renderer.tags = [name + ".guide_lines"]
  return cp

def dt_plots(X: ArrayLike, name: str, E: ArrayLike = [], d: int = 2, **kwargs):
  ''' Defines a sequence of plots (sp,hp,cp) for understanding directional transforms '''
  assert d == 2, "Only 2D supported for now"
  ## Signature plot
  sp = signature_plot(name, width=300,height=300)
  vline = Span(location=0, dimension='height', line_color='black', line_width=2.5)
  vline.tags = [name+".vline"]
  sp.add_layout(vline)

  ## Diagram/Halfplane plot
  hp = dgm_plot(name, width=300,height=300)
  hp_boxtool = _box_tool(hp, name)
  hp.add_tools(hp_boxtool)

  ## Circle plot 
  cp = circle_plot(name, X)
  line_src = ColumnDataSource({ 'x' : [0,1.5], 'y': [0,0] })
  # Arrow(end=NormalHead(fill_color="black"), x_start=0, y_start=0, x_end=0, y_end=0.7)
  # s1_arrow_renderer = cp.arrow()
  s1_arrow_renderer = cp.line(x="x", y="y", line_color="black", line_width=3, line_alpha=1.0, source=line_src)
  s1_arrow_renderer.tags = [name+".unit_vector"]

  ## Add level guides 
  # slope = Slope(gradient=gradient, y_intercept=y_intercept, line_color='orange', line_dash='dashed', line_width=3.5)
  # cp.line()  


  ## If edges passed in, render complex
  vr,er,tr = render_complex(cp, X, E=E, T=[])
  vr.tags = [name+".v_renderer"]
  er.tags = [name+".e_renderer"]
  tr.tags = [name+".t_renderer"]
  return sp, hp, cp

## Get data (+center and scale it)
D = mpeg7()
for name in D.keys(): D[name] = pht_preprocess_pc(D[name],nd=32)

## TODO: preprocess mpeg or fix somehow 
R_refl = np.array([[-1, 0], [0, 1]])
D[('turtle',1)] = D[('turtle',1)] @ R_refl

## The active data set global variables
ds_name = ""
X = np.empty(shape=(0,0)) 
E = np.empty(shape=(0,2))
sp, hp, cp = None, None, None     # (signature / half-plane / circle) plots
vr, lr, vline = None, None, None  # vertex renderer (cp), line renderer, vertical line (sp)
br = None                         # box renderer (hp)
gr = None                         # guide lines renderer (cp)

## Settings 
log_msgs = []
w,h = 300, 300
TOOLS = ["pan", "wheel_zoom", "reset"]
control_panel = column([], name="controls")

## Text support 
directions = Div(text="<b> Control Panel </b>")
# control_panel.children.append(directions)

## Add data picking options
DATA_OPTIONS = [str(k) for k in D.keys()]
pc_data_source = ColumnDataSource({ 'data_keys' : DATA_OPTIONS[:6] })
# def choose_dataset(attr: str, old: List, new: List):
#   print(old)
#   print(new)
#   pc_data_source.data['data_keys'] = new
# data_choice = MultiChoice(value=DATA_OPTIONS[0:3], options=DATA_OPTIONS, placeholder="Data set(s)")
# data_choice.on_change('value', choose_dataset)
# control_panel.children.append(data_choice)
# control_panel.children.append(Div(text="<strong>Local options</strong> </hr>")) # style={ 'margin-bottom': '1rem','margin-bottom':'1rem','border' : '1px', 'border-top': '1px solid rgba(0, 0, 0, 0.1)' }
CONTEXT_PLOTS = {}

## Get the directional-transform plots
dt_figs, control_tabs = Tabs(tabs=[]), Tabs(tabs=[])
for name in pc_data_source.data['data_keys']:
  # global ds_name, X, E, sp, hp, cp, lr, vr, vline, br
  ds_name = name
  X = D[eval(ds_name)]
  E = np.array(list(cycle_window(range(X.shape[0]))))
  sp,hp,cp = dt_plots(X, ds_name, E=E)
  CONTEXT_PLOTS[ds_name] = [sp,hp,cp] # experimental
  lr = cp.select(tags=[ds_name+".unit_vector"])
  vr = cp.select(tags=[ds_name+".v_renderer"])
  vline = sp.select(tags=[ds_name+".vline"])
  br = hp.select(tags=[ds_name+".boxes"]) # box renderer 

  ## Compute initial diagram
  dgm = ph0_lower_star(X @ np.array([1, 0]), E, max_death="max")
  dp_renderer = hp.select(tags=[ds_name+".dgm_points"])
  dp_renderer.data_source.data.update(
    x=dgm[:,0],
    y=dgm[:,1],
    alpha=np.repeat(1.0, dgm.shape[0]),
    theta=np.repeat(0.0, dgm.shape[0])
  )
  turbo_color = linear_cmap('theta', 'Turbo256', low=0, high=2*np.pi)

  lb,ub = min(dgm[:,0]), max(dgm[:,1])
  hp.x_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))
  hp.y_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))
  #dp_renderer.glyph.fill_color = {'field': 'theta', 'transform': Turbo256 }
  #dp_renderer.glyph.fill_color = turbo_color
  dt_figs.tabs.append(Panel(child=row(sp,hp,cp), title=ds_name))

# def render_complex(plot, X: ArrayLike, E: ArrayLike = [], T: ArrayLike = []):
#   X_src = ColumnDataSource({ 'x': X[:,0], 'y': X[:,1] , 'lambda': X[:,0] })
#   turbo_color = linear_cmap('lambda', 'Turbo256', low=-1.0, high=1.0) # DataSpec dict


  

## Every time a tab is clicked, switch the global variables
## NOTE: this is needed because .select(*) is SLOWWWWW
def ds_tab_cb(attr, old: int, new: int):
  global ds_name, X, E, sp, hp, cp, lr, vr, vline, br, gr
  print("clicked tab"+str(new))
  ds_name = dt_figs.tabs[new].title
  X = D[eval(ds_name)]
  E = np.array(list(cycle_window(range(X.shape[0]))))
  sp,hp,cp = dt_figs.tabs[new].child.children
  lr = cp.select(tags=[ds_name+".unit_vector"])
  vr = cp.select(tags=[ds_name+".v_renderer"])
  vline = sp.select(tags=[ds_name+".vline"])
  br = hp.select(tags=[ds_name+".boxes"])
  gr = cp.select(tags=[ds_name+".guide_lines"])

dt_figs.on_change('active', ds_tab_cb)

## Add directional-transformer slider control (universal)
S1_slider = Slider(name="Filter angle", start=0, end=2*np.pi, step=0.01, value=0.0)
def slider_cb(attr: str, old_theta: float, new_theta: float) -> None:
  # br = hp.select(tags=[ds_name+".boxes"])
  ## Get current active tab/data set
  v = np.array([np.cos(new_theta), np.sin(new_theta)])
  
  ## Update renderers
  lr.data_source.data['x'] = [0, v[0]*1.5]
  lr.data_source.data['y'] = [0, v[1]*1.5]
  vline.location = new_theta
  vr.data_source.data['lambda'] = X @ v

  ## Update any active guide renderers 
  if gr is not None:
    selected_box_ind = br.data_source.selected.indices
    if len(selected_box_ind) == 1:
      i = selected_box_ind[0]
      rx,ry,rw,rh = [br.data_source.data[k][i] for k in ['x', 'y', 'width', 'height']]
      update_guide_lines(gr, rx,ry,rw,rh, new_theta)

S1_slider.on_change("value", slider_cb)
# control_panel.children.append(S1_slider)

## Add vineyard controls
size_inp = Spinner(value=25, low=1, high=100, placeholder="Size of points", mode='int', width=int(w/5)) # title="Point Size",
nd_inp = NumericInput(value=32, low=1, high=512, placeholder="Number of directions", mode='int', width=int(w/5), align='end')
dgms_button = Button(label="DGMS", button_type="success", width=int(w/6))

# dgm_points = ColumnDataSource({ 'x': [], 'y': [], 'r_index': [], 'alpha': [] }) 
def dgm_cb(new):
  log_msgs.append('Dgm button selected.')
  F = list(rotate_S1(X, nd_inp.value, include_direction=False))
  dgms = [ph0_lower_star(f, E, max_death="max") for f in F]

  P = np.vstack([d for d in dgms])
  ds = np.array([d.shape[0] for d in dgms]).astype(int) # dgm sizes
  dp_r = hp.select(tags=[ds_name+".dgm_points"])
  dp_r.data_source.data = {
    'x' : P[:,0],
    'y' : P[:,1], 
    'theta' : (2*np.pi)*(np.repeat(list(range(len(ds))), ds)/nd_inp.value), # pht rotation index
    'alpha' : np.repeat(0.01, P.shape[0]), 
    'size' : np.repeat(size_inp.value, P.shape[0])
  }
  lb, ub = min(P.flatten()), max(P.flatten())
  # turbo_color = linear_cmap('theta', 'Turbo256', low=0, high=2*np.pi)
  # dp_renderer.glyph.fill_color = turbo_color
  # dp_renderer.glyph.update(fill_color=turbo_color)
  #hp.scatter(x='x', y='y', alpha='alpha', size=15, marker="dot", color=turbo_color, source=dgm_points)
  # hp
  hp.x_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))
  hp.y_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))

def pt_size_cb(attr: str, old: int, new: int) -> None:
  dp_r = hp.select(tags=[ds_name+".dgm_points"])
  dp_r.data_source.data['size'] = np.repeat(size_inp.value, len(dp_r.data_source.data['size']))
size_inp.on_change("value", pt_size_cb)

#dgm_button = Button(label="DGM", button_type="warning", width=int(w/6))
dgms_button.on_click(dgm_cb)
# control_panel.children.append(row(nd_inp, dgms_button, size_inp)) # Div(text="Point Size: ", height=500)

## Add the option to choose directional transforms
# menu = [("Euler", "euler"), ("Betti", "betti"), ("BettiNuclear", "betti_nuclear")]
dt_select = Select(value="Betti", options=["Euler", "Betti", "BettiNuclear", "BettiGeneric"], width=int(w/3))
# dt_dropdown = Dropdown(label="Transform", button_type="warning", menu=menu)
def dt_drop_cb(attr, old, new):
  print(f"Selected transform: {new} (from {old})")
  
  # print(event.item)
  # print(str(event))
# dt_dropdown.on_change('value', dt_drop_cb)
dt_select.on_change('value', dt_drop_cb)

## Add directional transform button
w_inp = Spinner(value=0.0, low=0, high=2.0, step=0.01, placeholder="omega", mode='float', width=int(w/5)) 
def dt_cb(new):
  # box_renderer = hp.select(tags=[ds_name+".boxes"])
  # box_source = box_renderer.data_source
  # box_source.selected
  # global_methods = ["Laplacian", "WeightedLaplacian"]
  # if dt_select.value == "Laplacian":
  ## First step: compute eigenvalues of weighted Laplacian on DT-family. Save as a sparse matrix. 
  ## Second step: To see effects of different smoothing methods, add a slider w/ a parameter eps 
  ## Third step: If user selects a set of rectilinear shapes + hits a compute button, replace the signature plot with smoothed versions of those 


  # else: 
  n_boxes = len(br.data_source.data['x'])
  if n_boxes > 0: 
    selected_ind = br.data_source.selected.indices
    selected_ind = selected_ind if len(selected_ind) > 0 else np.fromiter(range(n_boxes), dtype=int)
    F = list(rotate_S1(X, nd_inp.value, include_direction=False))
    R = []
    for rx,ry,rw,rh in zip(*[br.data_source.data[k] for k in ['x', 'y', 'width', 'height']]):
      a,b,c,d = rx-rw/2, rx+rw/2, ry-rh/2, ry+rh/2
      R.append([a,b,c,d])
    R = np.array(R)[selected_ind,:]
    log_msgs.append(f"DT transform of {R.shape[0]} being calculated")
    method: str = ""
    if dt_select.value == "BettiNuclear":
      method = "nuclear"
    elif dt_select.value == "BettiGeneric":
      method = "generic"
    else:
      method = "exact"
    log_msgs.append(f"method: {method}")
    M = lower_star_multiplicity(F, E, R, method=method, w=w_inp.value, epsilon=0.0)
    print(str(M))
    DT_X = (np.fromiter(range(len(F)), dtype=int)/len(F))*(2*np.pi)
    DT_Y = M.sum(axis=1).astype(int) if (method == "exact" or method == "rank") else M.sum(axis=1).astype(float)
    step_renderer = sp.select(tags=[ds_name+".step"])
    line_renderer = sp.select(tags=[ds_name+".line"])
    if method == "exact" or method == "rank":
      step_renderer.data_source.data = { 'x' : DT_X,  'y' : DT_Y }
      step_renderer.visible = True 
      line_renderer.visible = False
    else:
      print("making step renderer not visible")
      line_renderer.data_source.data = { 'x' : DT_X,  'y' : DT_Y }
      step_renderer.visible = False 
      line_renderer.visible = True
    sp.x_range.update(start=0, end=2*np.pi)
    rng = abs(max(DT_Y)-min(DT_Y))
    lb, ub = np.min(DT_Y)-0.10*rng, np.max(DT_Y)+0.10*rng
    if lb == ub: 
      lb, ub = lb - 1.0, ub + 1.0
    sp.y_range.update(start=lb, end=ub)
    log_msgs.append("Signature plotted")
  else: 
    print("No boxes created")
    M = []
      # log_msgs.append(''.join([str(box_glyph) for box_glyph in box_tool.renderers]))

dt_button = Button(label="DT", button_type="danger", width=int(w/6))
dt_button.on_click(dt_cb)
# control_panel.children.append(dt_button)

def show_circles_cb(active: bool):
  log_msgs.append("Tapped box")
  # if active:
  box_renderer = hp.select(tags=[ds_name+".boxes"])
  box_source = box_renderer.data_source
  selected_box_ind = box_source.selected.indices
  print("Selected indices: " + str(selected_box_ind))
  if len(selected_box_ind) == 1:
    # ar = cp.select(tags=[ds_name+".annuli"])
    i = selected_box_ind[0]
    rx,ry,rw,rh = [box_source.data[k][i] for k in ['x', 'y', 'width', 'height']]
    gr = cp.select(tags=[ds_name+".guide_lines"])
    gr.visible = True
    update_guide_lines(gr, rx, ry, rw, rh, S1_slider.value)
    
    # guideline_renderer
    # if len(gr) == 0:
    #   gr = cp.multi_line(XX, YY, line_width=1.5, color=bin_color([a,b,c,d], "turbo"), line_dash="dashed")
    #   gr.tags = [ds_name + ".guide_lines"]
    # else: 
    # print(gr.data_source.data)
    # gr.data_source.data = { 'xs' : XX, 'ys' : YY, 'line_color': bin_color([a,b,c,d], "turbo")}
    # print(gr.data_source.data)
    # a,b,c,d = rx-rw/2, rx+rw/2, ry-rh/2, ry+rh/2
    # print((a,b,c,d))
    # R = { 'x': [0,0], 'y': [0,0], 'ir' : [abs(b), abs(d)], 'or': [abs(a), abs(c)], 'alpha': [0.20,0.20], 'color': ['blue', 'green'] }
    # ar.data_source.data = R
    # ar.visible = True

    ## Other guide type
    # XY_p = np.c_[np.repeat([a,b,c,d],2), np.tile([-1.5, 1.5], 4)]
    # theta = S1_slider.value
    # RM = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # XY_p = (XY_p @ RM)
    # XX = np.array_split(XY_p[:,0],4)
    # YY = np.array_split(XY_p[:,1],4)

    
  else: 
    print("Turning off annuli")
    ar = cp.select(tags=[ds_name+".annuli"])
    ar.visible = False 




# hp_tap = hp.select(type=TapTool)
# hp_tap.on_event("tap", show_circles_cb)
toggle_box_circles = Toggle(label="Guides", button_type="success", width=int(w/5))
toggle_box_circles.on_click(show_circles_cb)
# control_panel.children.append(toggle_box_circles)


# ri: int = 0
# def animate_cb():
#   global ri 
#   NR = nd_inp.value
#   RI = np.array(dgm_points.data['r_index']) # rotation index 
#   A = np.repeat(0.10, len(RI))
#   ri = (ri + 1) % NR
#   # for ri in np.sort(np.unique(RI)):
#   print(f"{ri} out of ")
#   A[RI == ri] = 1.00
#   A[RI != ri] = 0.10
#   dgm_points.data.update(alpha=A) # data['alpha'] = A
#   # time.sleep(0.25) 
#   log_msgs.append(f"rotation index: {ri}")

# ## Add vines animation button
# vines_clicked: bool = False
# cb_object: int = 0
# def init_animate_cb(new):
#   global vines_clicked
#   global cb_object
#   ndp: int = len(dgm_points.data['x'])
#   if ndp > 0:
#     if not vines_clicked:
#       cb_object = curdoc().add_periodic_callback(animate_cb, 50)
#     else: 
#       curdoc().remove_periodic_callback(cb_object)
#       dgm_points.data.update(alpha=np.repeat(1.0, ndp))
#     vines_clicked = not(vines_clicked)
# animate_button = Button(label="Animate vines", button_type="danger", width=int(w/6))
# animate_button.on_click(init_animate_cb)
# control_panel.children.append(animate_button)


# control_panel.children.append(dt_select)

## Add log panel
log_div = Div(text='LOG:', width=w, height=h, style={"overflow-y": "scroll", "max-height" : "100px" }, name="Log")
def update_log():
  # print("Updating log")
  ms = ''.join([f"<li>{msg}</li>" for msg in reversed(log_msgs[-100:])])
  log_div.text = '<small><ul>{}</ul></small>'.format(ms)
# control_panel.children.append(log_div)

dt_dist = np.zeros(shape=(20,20))
dp = figure(width=w, height=h) # directional plot
dp.x_range.range_padding = dp.y_range.range_padding = 0
dp.image(image=[dt_dist], x=0, y=0, dw=10, dh=10, palette='Viridis256', level="image")
dp.visible = False


## Add (+) operator for boxes + weighting options 
## Keep option to display individual box multiplicities?
weight_select = Select(value="Uniform", options=["Uniform", "Lifetime"], width=int(w/3))

# Button to save or load all boxes
sl_all = CheckboxGroup(labels=["Apply to all"], active=[1])

srec_button = Button(label="Save boxes", button_type="default", width=int(w/4))
def save_recs_cb():
  global ALL_RECTANGLES
  br = hp.select(tags=[ds_name+".boxes"])
  R = []
  for rx,ry,rw,rh in zip(*[br.data_source.data[k] for k in ['x', 'y', 'width', 'height']]):
    a,b,c,d = rx-rw/2, rx+rw/2, ry-rh/2, ry+rh/2
    R.append([a,b,c,d])
  R = np.array(R)
  import copy 
  ALL_RECTANGLES = copy.deepcopy(dict(br.data_source.data.items())) ## save all rectangles to global 
  log_msgs.append("Saved rectangles: " + str(R))
  print(hp)
srec_button.on_click(save_recs_cb)

lrec_button = Button(label="Load boxes", button_type="default", width=int(w/5))
def load_recs_cb():
  # global ds_name, hp, br
  if sl_all.active[0] == 0: # cuz 0 is active  
    print("Loading boxes")
    print("Context plot keys: "+ str(list(CONTEXT_PLOTS.keys())))
    print(pc_data_source.data)
    for name in pc_data_source.data['data_keys']:
      print('here')
      sp_local,hp_local,cp_local = CONTEXT_PLOTS[name]
      print(hp_local)
      br_local = hp_local.select(tags=[name+".boxes"])
      if isinstance(ALL_RECTANGLES, dict):
        br_local.data_source.data = ALL_RECTANGLES
  else:
    # global ds_name, X, E, sp, hp, cp, lr, vr, vline, br
    ## bp,hp already loaded
    print(ds_name)
    print(hp)
    br_ = hp.select(tags=[ds_name+".boxes"])
    print(br_)
    # print("rectangles: " + str(ALL_RECTANGLES))
    # print("current data: " + str(br.data_source.data))
    if isinstance(ALL_RECTANGLES, dict):
      print(ALL_RECTANGLES)
      br_.data_source.data = ALL_RECTANGLES
  # br.update()
  # lb = min(br.data_source.data['x'])
  # sp.x_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))
  # sp.y_range.update(start=lb-0.1, end=ub+0.1, bounds=(lb-0.1, ub+0.1))
lrec_button.on_click(load_recs_cb)




dist_button = Button(label="Load boxes", button_type="default", width=int(w/4))
# for 

## Aligned plot 
ap = figure(width=w, height=h) # Aligned plot 
data_choice = MultiChoice(value=[], options=DATA_OPTIONS, placeholder="Data set(s) to compare")
def choose_comparison_cb(attr: str, old: List, new: List):
  from pbsig.signal import phase_align
  print(old)
  print(new)
  
  if len(new) > 0:
    ar_ds = ColumnDataSource({ 'x' : [], 'y': [] })
    dt_results = []
    for dsn in new: 
      idx = list(D.keys()).index(eval(dsn))
      print(idx)
      sp_,hp_,cp_ = dt_figs.tabs[idx].child.children
      sr_ = sp_.select(tags=[str(dsn)+'.step'])
      lr_ = sp_.select(tags=[str(dsn)+'.line'])
      
      if lr_.visible and len(lr_.data_source.data['y']) > 0:
        y = lr_.data_source.data['y']
      elif sr_.visible and len(sr_.data_source.data['y']) > 0: 
        y = sr_.data_source.data['y']
      else:
        y = []
      dt_results.append(y)
    
    ar.line(x="", y="", color="red", source=ar_ds)
        
    # sp.select(tags=)
    # ap.line()
  #ap.line()
data_choice.on_change('value', choose_comparison_cb)


# vy_tab = Panel(child=row(dgms_button, Div(text=" Num. Directions:  "), nd_inp), title="Vineyards")
# dt_tab = Panel(child=row(dt_button, Div(text="Transform: "), dt_select, w_inp), title="DT"
controls_tab = Panel(title="Controls", child=
  column(
    row(weight_select, srec_button, lrec_button, sl_all), 
    row(dgms_button, Div(text=" Num. Dir:  "), nd_inp),
    row(dt_button, Div(text="Transform: "), dt_select, w_inp)
  )
)
vis_tab = Panel(title="UI", child=
  column(
    row(toggle_box_circles, Div(text="Point size: "), size_inp), 
  )
)
log_tab = Panel(child=row(log_div, sizing_mode='stretch_width'), title = "Log")
# cmp_tab = Panel(child=row(data_choice), title="Compare")


controls = column(
  Div(text="<b> Control Panel </b>"),
  S1_slider, 
  # row(dgms_button, Div(text=" Num. Directions:  "), nd_inp),
  Tabs(tabs=[controls_tab, vis_tab, log_tab]), 
  #row(dt_button, Div(text="Transform: "), dt_select, w_inp),
  #row(toggle_box_circles, Div(text="Point size: "), size_inp),
  #row(weight_select, srec_button, lrec_button)
  width=400
)

## Form the final layout
## column(*) := "put the results in a column"
## row(*) := 
P = row(
  column(controls),
  column(
    row(dt_figs, dp), 
    row(ap)
  )
)

curdoc().add_periodic_callback(update_log, 200)
curdoc().add_root(P)
dt_figs.trigger("active", 0, 0)