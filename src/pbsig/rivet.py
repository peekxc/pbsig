from itertools import combinations
import os
import pathlib
import subprocess
from tempfile import NamedTemporaryFile
from typing import Iterable, Union, Callable
import numpy as np
import splex as sx


def bigraded_betti(
	S: sx.ComplexLike,
	f1: Callable,
	f2: Callable,
	p: int = 0,
	xbin: int = 15,
	ybin: int = 15,
	rivet_path: str = "~/rivet",
	verbose: int = 0,
	input_file: str = None,
	output_file: str = None,
	**kwargs,
):
	## Find RIVET CLI first
	rivet_path = os.path.expanduser(rivet_path)
	rivet_path = pathlib.Path(os.path.join(rivet_path, "rivet_console")).resolve()
	assert rivet_path.exists(), "Did not find file 'rivet_console'. Did you supply a valid 'rivet_path'?"
	rivet_path_str = str(rivet_path.absolute())

	## Prep the temporary file to hand off to rivet CLI
	tf = NamedTemporaryFile() if input_file is None else open(input_file, "w", encoding="utf-8")
	inp_file_str = "--datatype bifiltration\n"
	inp_file_str += "--xlabel density\n"
	inp_file_str += "--ylabel diameter\n"
	inp_file_str += "\n"
	for s, fs_1, fs_2 in zip(S, f1(S), f2(S)):
		inp_file_str += f"{' '.join([str(v) for v in s])} ; {fs_1} {fs_2}\n"
	with open(tf.name, "w") as inp_file:
		inp_file.write(inp_file_str)

	## Prep the rivet_console command
	output_tf = NamedTemporaryFile() if output_file is None else open(output_file, "w", encoding="utf-8")
	xreverse, yreverse = kwargs.get("xreverse", False), kwargs.get("yreverse", False)
	rivet_cmd = [rivet_path_str, inp_file.name]
	rivet_cmd += [
		"--betti",
		"--homology",
		str(p),
		"--xbins",
		str(xbin),
		"--ybins",
		str(ybin),
	]
	rivet_cmd += ["--xreverse" if xreverse else ""]
	rivet_cmd += ["--yreverse" if yreverse else ""]
	rivet_cmd += [">", output_tf.name]

	## Call rivet_console
	if verbose > 0:
		print(f"Calling rivet CLI with command: '{' '.join(rivet_cmd)}'")
	subprocess.run(" ".join(rivet_cmd), shell=True, check=True)
	# assert res == 0, "RIVET subprocess did not finish status code {res} "

	## Process the output file + return
	with open(output_tf.name) as f:
		rout = [line.rstrip("\n") for line in f]
	assert len(rout) > 0, "RIVET subprocess did not write any output."
	xi, yi, di, bi = [rout.index(key) for key in ["x-grades", "y-grades", "Dimensions > 0:", "Betti numbers:"]]
	b0, b1, b2 = [rout.index(key) for key in ["xi_0:", "xi_1:", "xi_2:"]]
	xg = np.array([eval(x) for x in rout[(xi + 1) : yi] if len(x) > 0])
	yg = np.array([eval(x) for x in rout[(yi + 1) : di] if len(x) > 0])

	def gen_tuples(L: Iterable, value: bool = True):
		for x in filter(lambda x: len(x) > 0, L):
			i, j, val = tuple(eval(x))
			yield (xg[i], yg[j], val) if value else (i, j, val)

	b_type = [("x", "f4"), ("y", "f4"), ("value", "i4")]
	betti_info = {}
	betti_info["x-grades"] = xg
	betti_info["y-grades"] = yg
	betti_info["hilbert"] = np.fromiter(gen_tuples(rout[(di + 1) : bi]), dtype=b_type)
	betti_info["0"] = np.fromiter(gen_tuples(rout[(b0 + 1) : b1]), dtype=b_type)
	betti_info["1"] = np.fromiter(gen_tuples(rout[(b1 + 1) : b2]), dtype=b_type)
	betti_info["2"] = np.fromiter(gen_tuples(rout[(b2 + 1) :]), dtype=b_type)
	return betti_info


def anchors(S: Union[np.ndarray, dict]):
	"""Computes the anchors of the set of bigraded Betti numbers S"""
	if isinstance(S, dict):
		b0 = np.c_[S["0"]["x"], S["0"]["y"]]
		b1 = np.c_[S["1"]["x"], S["1"]["y"]]
		S = np.vstack((b0, b1))
	assert isinstance(S, np.ndarray), "'S' must be the set of Betti numbers."
	anchors = set()
	for p, q in combinations(S, 2):
		if not (np.all(p <= q)) and (np.any(p < q) or p[0] == q[0] or p[1] == q[1]):
			lub = np.array([p, q]).max(axis=0)  # lowest upper bound
			anchors.add(tuple(lub))
	anchors = np.array(list(anchors), dtype=float)
	return anchors


def push_map(X: np.ndarray, a: float, b: float) -> np.ndarray:
	"""Projects set of points X in the plane onto the line given by f(x) = ax + b

	Returns:
					(x,y,f) = projected points (x,y) and the distance to the lowest projected point on the line.
	"""
	n = X.shape[0]
	out = np.zeros((n, 3))
	for i, (x, y) in enumerate(X):
		tmp = ((y - b) / a, y) if (x * a + b) < y else (x, a * x + b)
		dist_to_end = np.sqrt(tmp[0] ** 2 + (b - tmp[1]) ** 2)
		out[i, :] = np.append(tmp, dist_to_end)
	return out


def figure_betti(betti: dict, **kwargs):
	"""Creates a figure of the dimension function + the bigraded Betti numbers"""
	from bokeh.plotting import figure
	from pbsig.vis import rgb_to_hex
	BI = betti
	h = BI["hilbert"]
	xg, yg = BI["x-grades"], BI["y-grades"]
	x_step, y_step = np.abs(np.diff(xg)[0]), np.abs(np.diff(yg)[0])
	box_intensity = ((1.0 - (h["value"] / max(h["value"]))) * 255).astype(int)
	box_color = np.repeat(box_intensity, 3).reshape(len(h["value"]), 3).astype(np.uint8)
	
	from bokeh.models import ColumnDataSource
	r_source = ColumnDataSource(data=dict(
		x = h["x"] + x_step / 2.0, 
		y = h["y"] + y_step / 2.0, 
		fill_color = box_color,
		value = h["value"]
	))
	r_tooltips = [
		# ("(x,y)", "(@x,@y)"),
		("dim", "@value")
	]

	## Make the figure
	fig_kwargs = dict(width=250, height=250) | kwargs
	p = figure(**fig_kwargs, tooltips = r_tooltips)
	rect_r = p.rect(
		x='x', y='y', fill_color= 'fill_color',
		width=x_step, height=y_step,
		fill_alpha=0.25, line_color="black", line_alpha=0.0, line_width=0.0, 
		source = r_source
	)
	p.hover.renderers = [rect_r] # hover only for rectangles
	p.scatter(BI["0"]["x"], BI["0"]["y"], size=12, color=rgb_to_hex([52, 209, 93, int(0.75 * 255)]))
	p.scatter(BI["1"]["x"], BI["1"]["y"], size=12, color=rgb_to_hex([212, 91, 97, int(0.75 * 255)]))
	p.scatter(BI["2"]["x"], BI["2"]["y"], size=12, color=rgb_to_hex([254, 255, 140, int(0.75 * 255)]))
	return p


