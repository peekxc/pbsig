
from multiprocessing.sharedctypes import Value
import numpy as np
from numpy.typing import ArrayLike
from typing import *
from .color_constants import COLORS
# Largely from: https://bsouthga.dev/posts/color-gradients-with-python

def colors_to_hex(x):
	''' Given an iterable of color names, rgb values, or a hex strings, returns a list of the corresponding hexadecimal color representation '''
	def convert_color(c):
		if isinstance(c, str):
			return(c if c[0] == "#" else rgb_to_hex(COLORS[c]))
		elif len(c) == 3:
			return(rgb_to_hex(c))
		else: 
			raise ValueError("Invalid input detected")
	return(np.array([convert_color(c) for c in x]))


# hex_to_rgb(colors_to_hex(pt_col*255))
from matplotlib.colors import to_hex

def hex_to_rgb(hex_str: Union[List, str]):
	''' "#FFFFFF" -> [255,255,255] '''
	# to_rgb(colors_to_hex(pt_col*255)[0])
	# Pass 16 to the integer function for change of base
	if isinstance(hex_str, str):
		return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]	
	else:
		return np.array([hex_to_rgb(h) for h in hex_str], dtype=np.uint8)


def rgb_to_hex(rgb):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  rgb = [int(x) for x in rgb]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in rgb])


def color_dict(gradient):
  ''' Converts a list of RGB lists to a dictionary of colors in RGB and hex form. '''
  return {"hex":[rgb_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def _linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
	''' returns a gradient list of (n) colors between
		two hex colors. start_hex and finish_hex
		should be the full six-digit color string,
		inlcuding the number sign ("#FFFFFF") '''
	# Starting and ending colors in RGB form
	s = hex_to_rgb(start_hex)
	f = hex_to_rgb(finish_hex)
	# Initilize a list of the output colors with the starting color
	RGB_list = [s]
	# Calculate a color at each evenly spaced value of t from 1 to n
	for t in range(1, n):
		# Interpolate RGB vector for color at the current value of t
		curr_vector = [
			int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
			for j in range(3)
		]
		# Add it to our list of output colors
		RGB_list.append(curr_vector)
	return color_dict(RGB_list)


def linear_gradient(colors, n):
	''' returns a list of colors forming linear gradients between
			all sequential pairs of colors. "n" specifies the total
			number of desired output colors '''
	colors = colors_to_hex(colors)
	# The number of colors per individual linear gradient
	n_out = int(float(n) / (len(colors) - 1))
	# returns dictionary defined by color_dict()
	gradient_dict = _linear_gradient(colors[0], colors[1], n_out)

	if len(colors) > 1:
		for col in range(1, len(colors) - 1):
			next = _linear_gradient(colors[col], colors[col+1], n_out)
			for k in ("hex", "r", "g", "b"):
				gradient_dict[k] += next[k][1:] # Exclude first point to avoid duplicates
	return gradient_dict

def hist_equalize(x, number_bins=100000):
	h, bins = np.histogram(x.flatten(), number_bins, density=True)
	cdf = h.cumsum() # cumulative distribution function
	cdf = np.max(x) * cdf / cdf[-1] # normalize
	return(np.interp(x.flatten(), bins[:-1], cdf))

## What should this do? 
## Something like this for equalization: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
def scale_interval(x: Iterable, scaling: str = "linear", min_x: Optional[float] = None, max_x: Optional[float] = None, **kwargs):
	"""
	Scales 
	Re-scales the values of an input iterable 'x' using the rule 'scaling' in such a way that: 
		1.  
	"""
	
	## Convert to numpy array 
	x = np.asarray(list(x))
	if min_x is None and max_x is None and len(np.unique(x)) == 1:
		return 

	## Detect if scaled output interval is desired; if not use existing range
	out_min = float(np.min(x)) if min_x is None else float(min_x)
	out_max = float(np.max(x)) if max_x is None else float(max_x)
	assert isinstance(out_min, float) and isinstance(out_max, float)
	
	## Normalize to unit interval
	x = (x-out_min)/(out_max-out_min)
	
	## Scale the values based on selected scaling
	if scaling == "linear":
		x = x
	elif scaling == "logarithmic":
		# x = scale_interval(np.log(x + 1.0), "linear", min_x=0.0, max_x=np.log(2))
		x = np.log((x*1000)+1)/np.log(1001)
		# (x-np.log(1))/(np.log(2000)-np.log(1))
		# np.log((np.linspace(0,1,10,endpoint=True)+1)*1000)/np.log(2000)
		# (np.log((x+1)*1000)-np.log(1000))/np.log(2000)
	elif scaling == "equalize":
		sh = x.shape
		x = hist_equalize(x, **kwargs).reshape(sh)
	else:
		raise ValueError(f"Unknown scaling option '{scaling}' passed. Must be one of 'linear', 'logarithmic', or 'equalize'. ")
	
	## Finally, re-translate and scale entries x to [min_x, max_x]. 
	## If not supplied, this re-scales input values to their original range
	out = (out_min + x*(out_max-out_min)) if scaling == "linear" else scale_interval(x, "linear", out_min, out_max)
	return(out)

def bin_color(x: Iterable, color_pal: Optional[Union[List, str]] = 'viridis', lb: Optional[float] = None, ub: Optional[float] = None, **kwargs):
	''' Bins non-negative values 'x' into appropriately scaled bins matched with the given color range. '''
	if isinstance(color_pal, str):
		import bokeh
		bokeh_palettes = { p.lower() : p for p in dir(bokeh.palettes) if p[0] != "_" }
		color_pal = getattr(bokeh.palettes, bokeh_palettes[color_pal.lower()])
		if isinstance(color_pal, Callable):
			color_pal = np.c_[hex_to_rgb(color_pal(255))/255.0, np.ones(255)]
		elif isinstance(color_pal, dict):
			x_discrete = x.astype(np.int16)
			x_class, x_ind = np.unique(x_discrete, return_inverse=True)
			discrete_cp  = color_pal[max(len(x_class), 3)] 
			color_pal = np.c_[hex_to_rgb(discrete_cp)/255.0, np.ones(len(discrete_cp))]
			# discrete_rgb = hex_to_rgb([discrete_cp[i] for i in x_ind])
			# color_pal = np.hstack((discrete_rgb/255.0, np.ones(len(x_class))[:,np.newaxis]))
		else: 
			raise ValueError("If color_pal is a string, it must be a dict- or function-valued palette in bokeh.palettes")
		# color_pal = np.c_[np.array(color_pal(255)), np.ones(255)]
		# col = cm.get_cmap(color_pal)
		# color_pal = [col(i) for i in range(0, 255)]
		# color_pal = cm.get_cmap(color_pal).colors
	# else: 
	# 	raise ValueError("Unknown color map")
	# x = scale_interval(x, **kwargs)
	lb = float(np.min(x)) if lb is None else float(lb)
	ub = float(np.max(x)) if ub is None else float(ub)
	ind = np.digitize(np.clip(x, a_min=lb, a_max=ub), bins=np.linspace(lb, ub, len(color_pal)))
	ind = np.minimum(ind, len(color_pal)-1) ## bound from above
	ind = np.maximum(ind, 0)								## bound from below
	return np.asarray(color_pal)[ind]
	# return(hex_to_rgb(np.asarray(color_pal)[ind]))


# From: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
			A 2D numpy array of shape (M, N).
	row_labels
			A list or array of length M with the labels for the rows.
	col_labels
			A list or array of length N with the labels for the columns.
	ax
			A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
			not provided, use current axes or create a new one.  Optional.
	cbar_kw
			A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
			The label for the colorbar.  Optional.
	**kwargs
			All other arguments are forwarded to `imshow`.
	"""
	import matplotlib.pyplot as plt
	import matplotlib
	if ax is None: ax = plt.gca()
	if cbar_kw is None: cbar_kw = {}

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	# Show all ticks and label them with the respective list entries.
	ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
	ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

	# Turn spines off and create white grid.
	ax.spines[:].set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)
	return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
	"""
	A function to annotate a heatmap.

	Parameters
	----------
	im
			The AxesImage to be labeled.
	data
			Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
			The format of the annotations inside the heatmap.  This should either
			use the string format method, e.g. "$ {x:.2f}", or be a
			`matplotlib.ticker.Formatter`.  Optional.
	textcolors
			A pair of colors.  The first is used for values below a threshold,
			the second for those above.  Optional.
	threshold
			Value in data units according to which the colors from textcolors are
			applied.  If None (the default) uses the middle of the colormap as
			separation.  Optional.
	**kwargs
			All other arguments are forwarded to each call to `text` used to create
			the text labels.
	"""
	import matplotlib
	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	threshold = im.norm(threshold) if threshold is not None else im.norm(data.max())/2
	
	# Set default alignment to center, but allow it to be overwritten by textkw.
	kw = dict(horizontalalignment="center", verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)

	return texts