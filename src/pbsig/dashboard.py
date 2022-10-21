# dashboard.py
import numpy as np 
import bokeh 
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, show, curdoc
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.models import Range1d, ColumnDataSource, BoxEditTool, Model, Annulus, Plot, AnnularWedge, Slider, Span, Panel, Tabs, Button, Div, Line, PolyAnnotation, Rect
from bokeh.core.property.color import Color 
from bokeh.transform import linear_cmap

# output_notebook(verbose=False, hide_banner=True)

log_msgs = []
w,h = 300, 300
TOOLS = ["pan", "wheel_zoom", "reset"]

## Setup halfplane on [lb,ub]
lb, ub = 0.0, 1.0
hp = figure(title="Barcode", x_axis_label="birth", y_axis_label="death", width=w, height=h, tools=TOOLS+['tap'])
hp.line([lb,ub], [lb,ub], color="black", line_width=1)
hp.toolbar.logo = None
hp.y_range = Range1d(lb, ub, bounds=(lb, ub))
hp.x_range = Range1d(lb, ub, bounds=(lb, ub))
lhp = PolyAnnotation(fill_color="gray",fill_alpha=0.95,xs=[0, 1, 1, 0],ys=[0, 0, 1, 0])
lhp.level = 'underlay'
hp.add_layout(lhp)


## Setup box renderer
box_data = {'x': [0.30], 'y': [0.70], 'width': [0.10], 'height': [0.10], 'alpha': [0.5]}
# box_data = {'x': [], 'y': [], 'width': [], 'height': [], 'alpha': []}
box_source = ColumnDataSource(data=box_data)
box_renderer = hp.rect('x', 'y', 'width', 'height', alpha='alpha', source=box_source, fill_color="gray", fill_alpha=0.30, line_color=None)
selected_box = Rect(fill_color="firebrick", fill_alpha=0.90, line_color="black")
unselected_box = Rect(fill_color="gray", fill_alpha=0.30, line_color=None)
box_renderer.selection_glyph = selected_box
box_renderer.nonselection_glyph = unselected_box
box_tool = BoxEditTool(renderers=[box_renderer])
hp.add_tools(box_tool)

from typing import List 
from bokeh.events import DoubleTap, Tap, MouseEnter, ButtonClick, SelectionGeometry, Press
def my_cb2(attr: str, old: List[int], new: List[int]) -> None:
  log_msgs.append(f"box {str(new)} selected")
box_source.selected.on_change('indices', my_cb2)
# hp.on_event(MouseEnter, my_cb)


## Signature plot 
theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
# r"\[\beta_p(K)\]"
sp = figure(title="", x_axis_label="Angle", y_axis_label=None, width=w, height=h, tools=TOOLS+['xpan'])
sp.line(theta, np.sin(theta))
sp.toolbar.logo = None

## Circle plot 
#plot.add_glyph(source, glyph)
#xaxis = LinearAxis()plot.add_layout(xaxis, 'below')
cp = figure(title=None, width=300, height=300, min_border=0, toolbar_location=None)
cp.y_range = Range1d(-1.5,1.5, bounds=(-1.5,1.5))
cp.x_range = Range1d(-1.5,1.5, bounds=(-1.5,1.5))
S1 = cp.annulus(x=0, y=0, inner_radius=1.0, outer_radius=1.05, fill_color="#7fc97f")
# S1.glyph.on_event(MouseEnter, my_cb)
# S1.glyph.on_event(DoubleTap, my_cb)
# S1.glyph.on_event(Tap, my_cb)
# S1.glyph.on_event(ButtonClick, my_cb)
# S1.glyph.on_event(SelectionGeometry, my_cb)
# S1.glyph.on_event(Press, my_cb)

## Add angle slider
viridis_color = linear_cmap('start_angle', 'Turbo256', low=0, high=2*np.pi)
angle_src = ColumnDataSource({ 'start_angle': [0], 'end_angle': [np.pi/16] })
S1_slider = Slider(start=0, end=2*np.pi, step=0.01, value=0.0)

## 
vline = Span(location=0, dimension='height', line_color='black', line_width=2.5)
line_source = ColumnDataSource(dict(x=[0,1], y=[0,0]))
s1_arrow = cp.line(x="x", y="y", line_color="black", line_width=3, line_alpha=1.0, source=line_source)
# cp.add_glyph(s1_arrow)

def slider_callback(attr: str, old_theta: float, new_theta: float) -> None:
  # log_msgs.append(new)
  # log_msgs.append(str(s1_arrow.glyph.x))
  line_source.data['x'] = [0, np.cos(new_theta)]
  line_source.data['y'] = [0, np.sin(new_theta)]
  # log_msgs.append(str(line_source.data['x']))
  vline.location = new_theta
S1_slider.on_change("value", slider_callback)
sp.add_layout(vline)

## Controls panel
info_div = Div(text="<i>Betti Hash</i> panel info div", width=w, height=50)

compute_button = Button(label="Compute hashes", button_type="primary", width=int(w/3))
def compute_cb(new):
  log_msgs.append('Compute button selected.')
  log_msgs.append(', '.join([str(c) for c in box_source.data['x']]))
  # log_msgs.append(''.join([str(box_glyph) for box_glyph in box_tool.renderers]))
compute_button.on_click(compute_cb)

# direction_button = Button(label="Update direction", button_type="success", width=int(w/3))
# def direction_callback(new):
#   log_msgs.append('Radio button option ' + str(new) + ' selected.')



## Add log panel
log_div = Div(text='LOG:', width=w, height=h, style={"overflow-y": "scroll", "max-height" : "300px" }, name="Log")
def update_log():
  ms = ''.join([f"<li>{msg}</li>" for msg in reversed(log_msgs)])
  log_div.text = '<small><ul>{}</ul></small>'.format(ms)
  
# log_button.on_click(update_log)


# P = column(
#   row(hp, sp, cp, column(S1_slider, compute_button, log_button)), 
#   log_div
# )
P = row(hp, sp, cp, column(S1_slider, compute_button, row(log_div, name="Log", sizing_mode="stretch_both"), name="controls"))
# show(P, notebook_handle=True)
curdoc().add_periodic_callback(update_log, 200)
curdoc().add_root(P)

# bokeh serve dashboard.py --dev
# bkapp(doc)

#doc.add_root(model)


# show(bkapp)

  # show(curdoc())
  # 

  # c = Color()
  # c.prop = "#FFA50019"
  # sliders = column(amp, freq, phase, offset)  #wc = cp.annular_wedge(x=0, y=0, inner_radius=0.95, outer_radius=1.10, start_angle="start_angle", end_angle="end_angle", fill_color=viridis_color, source=angle_src)# source, wc 
  

  #S1_slider.js_link('value', wc, 'start_angle')
  # S1_slider.js_link('value', vline, 'location')
    #b1 = hp.quad(top=[0.80], bottom=[0.60], left=[0.20], right=[0.4], color="#FFA500")