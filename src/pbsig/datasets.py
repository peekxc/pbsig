import os
import io
import numpy as np
import pickle 

from os.path import exists
from scipy.spatial.distance import pdist, cdist, squareform
from typing import * 
from numpy.typing import ArrayLike
from pathlib import Path
from splex import filtration

# Package relative imports 
from .utility import *
from .combinatorial import *
from .simplicial import *
from .shape import simplify_outline
# from .__init__ import _package_data

from scipy.io import loadmat

_ROOT = os.path.abspath(os.path.dirname(__file__))
def _package_data(path: str = ""):
  return os.path.join(_ROOT, 'data', path)

def random_lower_star(n: int = 50, v: list = [0,1]):
  """ 
  Generates a random lower-star filtration from a delaunay complex over n random points in the plane. 
  The sublevel sets (lower-star) are given by the projection of each simplex onto the unit-vector v
  """
  assert len(v) == 2, "Invalid unit vector"
  v = np.array(v)/np.linalg.norm(v)
  X = np.random.uniform(size=(n,2))
  fv = X @ np.array(v)
  S = delaunay_complex(X)
  K = filtration(S, f=lambda s: max(fv[s]))
  return X, K

def shrec20():
  import pywavefront
  scene = pywavefront.Wavefront('/Users/mpiekenbrock/Downloads/SHREC20b_lores/models/camel_b.obj', collect_faces=True)
  triangles = scene.mesh_list[0].faces
  nv = len(scene.vertices)
  S = simplicial_complex(triangles)
  return scene.vertices, S 


def noisy_circle(n: int = 32, n_noise: int = 10, perturb: float = 0.05, r: float = 0.20):
  """Samples points around the unit circle
  Parameters: 
    n: number of points around the main circle 
    n_noise: number of additional noise points to add outside the _n_ points along the circle
    perturb: the value p such that that polar coordinates are pertubed by 1 +/- p
    r: minimum Hausdorff distance from the main circle the rejection sampler will consider a point as 'noise'
  """
  theta = np.linspace(0, 2*np.pi, n, endpoint=False)
  perturb_theta = np.random.uniform(size=n, low=1-perturb, high=1+perturb)
  circle = np.c_[np.sin(theta*perturb_theta), np.cos(theta*perturb_theta)]
  circle *= np.random.uniform(size=circle.shape, low=1-perturb, high=1+perturb)
  def _reject_sampler(n: int, r: float):
    pts = []
    while len(pts) < n:
      pt = np.random.uniform(size=(2,), low=-1-perturb, high=1+perturb)
      Y = circle if len(pts) == 0 else np.vstack([circle, np.array(pts)])
      if all([np.linalg.norm(pt - y) > r for y in Y]):
        pts.append(pt)
    return np.array(pts)
  noise = _reject_sampler(n_noise, r) if n_noise > 0 else np.empty(shape=(0, 2))
  X = np.vstack((circle, noise))
  return X 

def random_function(n_extrema: int = 10, n_pts: int = 50, order_penalty = 0.10, walk_distance = 0.05, eps: float = None, plot: bool = False):
  from csaps import CubicSmoothingSpline
  from scipy.optimize import golden
  walk_distance = (0.50 + walk_distance)
  movement = np.random.uniform(size=n_pts, low=-walk_distance, high=walk_distance)
  position = np.cumsum(movement)+0.50
  x = np.linspace(0, 1, len(position))
  def objective_f(eps: float) -> float:
    f = CubicSmoothingSpline(x, position, smooth=1.0-eps)
    roots = f.spline.derivative(1).roots() # jerk = f.spline.derivative(2)(roots) 
    n_crit = len(roots)
    return abs(n_crit - n_extrema) + order_penalty * f.spline.order
  eps_opt = golden(objective_f, brack=(0, 1e-3)) if eps is None else eps
  f = CubicSmoothingSpline(x, position, smooth=1-eps_opt)
  if plot: 
    from pbsig.vis import figure, show
    dom = np.linspace(0, 1, 1500)  
    roots = f.spline.derivative(1).roots()
    roots = roots[np.logical_and(roots >= 0, roots <= 1)]
    p = figure(width=300, height=250)
    p.line(dom, f(dom), color="black")
    p.scatter(roots, f(roots), size=6, color='red')
    show(p)
    return f, p
  return f



def animal_svgs():
  """
  Loads a list of animal svgs
  """
  from svgpathtools import svg2paths
  from svgpathtools.path import Path
  from pbsig import data as package_data_mod
  data_path = package_data_mod.__path__._path[0] + "/animal_svgs"
  SVG_fns = [file for file in os.listdir(data_path) if file.endswith('.svg')]
  SVG_paths = [svg2paths(data_path + "/" + svg)[0][0] for svg in SVG_fns]
  assert all(isinstance(p, Path) for p in SVG_paths), "Invalid set of SVGs"
  # SVG_paths[6][3]
  return(dict(zip([os.path.splitext(fn)[0] for fn in SVG_fns], SVG_paths)))

def letter_image(text, font: Optional[str] = ["Lato-Bold", "OpenSans", "Ostrich", "Oswald", "Roboto"], nwide=51):
  from PIL import Image, ImageDraw, ImageFont, ImageFilter
  """Generates a grayscale image of a letter in a given font.
  """
  base_dir = _package_data('fonts')
  fonts = ['lato-bold','opensans','ostrich', 'oswald','roboto']
  fonts_fn = ["Lato-Bold", "OpenSans-Bold", "ostrich-regular", "Oswald-Bold", "Roboto-Bold"]
  if font == ["Lato-Bold", "OpenSans", "Ostrich", "Oswald", "Roboto"]:
    font_path = base_dir + '/Lato-Bold.ttf'
  elif isinstance(font, str) and font.lower() in fonts:
    font_path = base_dir +'/' + fonts_fn[fonts.index(font.lower())] + '.ttf'
  else:
    # TODO: allow font paths 
    raise ValueError("invalid font given")

  ## Draw the RGBA image 
  image = Image.new("RGBA", (nwide,nwide), (255,255,255))
  draw = ImageDraw.Draw(image)
  path = Path(font_path).resolve()
  font = ImageFont.truetype(str(path), 40)

  #w,h = draw.textsize(text, font=font)
  w,h = draw.textbbox(xy=(0,0), text=text, font=font)[2:4]
  wo = int((nwide-w)/2)
  ho = int((nwide-h)/2)-int(11/2)
  draw.text((wo, ho), text, fill="black", font=font)

  pixels = np.asarray(image.convert('L'))
  return(pixels)

def letters():
  pass

def freudenthal_image(image: ArrayLike, threshold: int = 200):
  """
  Given an grayscale image and a threshold between [0, 255], returns the 'Freundenthal triangulation'
  of the embedding given by keeping every pixel below the supplied threshold as a vertex and connecting 
  adjacent vertices with edges + triangles. 
  """
  assert threshold >= 0 and threshold <= 255
  nwide = image.shape[1]

  ## Get position of vertices on pixel-grid less than threshold
  v_pos = np.column_stack(np.where(image < threshold))
  v_pos = np.fliplr(v_pos)
  v_pos[:,1] = nwide-v_pos[:,1]
  
  ## Connect adjacent vertices w/ edges 
  e_ind = np.where(pdist(v_pos, 'chebychev') == 1)[0]

  ## Restrict edge set to be 'freundenthal' edges
  def is_freundenthal(e):
    i,j = unrank_C2(e, v_pos.shape[0])
    x1, x2 = v_pos[i,0], v_pos[j,0]
    y1, y2 = v_pos[i,1], v_pos[j,1]
    return(((x1 < x2) and (y1 == y2)) or ((x1 == x2) and (y1 < y2)) or ((x1 < x2) and (y1 > y2)) or ((x1 == x2) and (y1 > y2)))
  e_fr = np.array(list(filter(is_freundenthal, e_ind)))
  E = np.array([unrank_C2(e, v_pos.shape[0]) for e in e_fr])
  
  ## Expand 1-skeleton to obtain triangles
  T = expand_triangles(v_pos.shape[0], E)
  
  ## Translate + scale
  center = np.array([
    np.min(v_pos[:,0]) + (np.max(v_pos[:,0])-np.min(v_pos[:,0]))/2,
    np.min(v_pos[:,1]) + (np.max(v_pos[:,1])-np.min(v_pos[:,1]))/2
  ])
  X = scale_diameter(v_pos - center) 
  return(X, simplicial_complex(T))

def open_tar(fn: str, mode: str = "r:gz"):
  import tarfile
  tar = tarfile.open(fn, mode)
  for member in tar.getmembers():
    yield member
    # f = tar.extractfile(member)
    # if f is not None:
    #   yield f

def _largest_contour(img: ArrayLike, threshold: int = 180):
  import cv2
  _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = (contours[np.argmax([len(c) for c in contours])])
  S = contours[:,0,:]
  return(S)


def mpeg7(contour: bool = True, simplify: int = 150, which: str = 'default', shape_nums = "all", return_X_y: bool = False):
  from PIL import Image, ImageDraw, ImageFont, ImageFilter
  shape_nums = range(21) if isinstance(shape_nums, str) and shape_nums == "all" else shape_nums
  base_dir = _package_data('mpeg7')
  mpeg7 = []
  _all_shapes = ['Bone', 'Comma', 'Glas', 'HCircle', 'Heart', 'Misk', 'apple', 'bat', 
       'beetle', 'bell', 'bird', 'bottle', 'brick', 'butterfly', 
       'camel', 'car', 'carriage', 'cattle', 'cellular_phone', 'chicken',
       'children', 'chopper', 'classic', 'confusions', 'confusions.eps',
       'confusions.fig', 'crown', 'cup', 'deer', 'device0', 'device1',
       'device2', 'device3', 'device4', 'device5', 'device6', 'device7',
       'device8', 'device9', 'dog', 'elephant', 'face', 'fish',
       'flatfish', 'fly', 'fork', 'fountain', 'frog', 'guitar', 'hammer',
       'hat', 'horse', 'horseshoe', 'jar', 'key', 'lizzard', 'lmfish',
       'octopus', 'pencil', 'personal_car', 'pocket', 'rat', 'ray',
       'sea_snake', 'shapedata', 'shapedata.eps', 'shapedata.fig', 'shoe',
       'spoon', 'spring', 'stef', 'teddy', 'tree', 'truck', 'turtle',
       'watch']
  _all_shapes = [s.lower() for s in _all_shapes]
  default_shapes = list(sorted([
    "turtle", "watch", "bird", "bone", "bell", "bat", "beetle", "butterly", 
    "car", "cup", "teddy", "spoon", "shoe", "ray", "sea_snake", "personal_car", 
    "key", "horse", "hammer", "frog", "fork", "flatfish", "elephant"
  ]))
  if isinstance(which, str) and which == "default":
    shape_types = default_shapes
  elif isinstance(which, str) and which == "all":
    shape_types = _all_shapes
  else:
    assert all([s in _all_shapes for s in which]), "Invalid set of shapes given"
    shape_types = which
  if simplify == 150 and contour == True:
    mpeg7 = pickle.load(open(base_dir + "/mpeg_small.pickle", "rb"))
    return { k : v for k,v in mpeg7.items() if k[0] in shape_types }
  # shape_nums = range(21) #[1,2,3,4,5,6,7,8,9,10,11,12] #3,4,5
  normalize = lambda X: (X - np.min(X))/(np.max(X)-np.min(X))*255
  dataset = {}
  for st in shape_types:
    for sn in shape_nums:
      img_dir = base_dir + f"/{st}-{sn}.gif"
      if not exists(img_dir):
        continue
      im = Image.open(img_dir)
      img_gray = normalize(np.array(im)).astype(np.uint8)
      if contour:
        S = _largest_contour(img_gray)
        S = _largest_contour(255-img_gray) if len(S) <= 4 else S # recompute negative if bbox was found
        assert len(S) > 4
        S = simplify_outline(S, simplify) if simplify > 0 else S 
        dataset[(st, sn)] = S # np.array([l.start for l in S]) 
      else: 
        dataset[(st, sn)] = img_gray
  if return_X_y:
    labels = [k[0] for k in dataset.keys()]
    classes, y = np.unique(labels, return_inverse=True)
    X_mat = np.array([np.ravel(emb) for emb in dataset.values()])
    return X_mat, y
  else:
    return(dataset)
  

def pose_meshes(simplify: int = None, which: Union[str, Iterable] = "default", shape_nums: Union[str, Iterable] = "all"):
  """3-D mesh data of animals / humans in different poses.

  Parameters: 
    simplify: number of triangles desired from the loaded mesh. By default no simplification is done.  
    which: which shapes to load. Defaults to ["camel", "cat", "elephant", "flamingo"]. 
    shape_nums: which poses to use (most shapes have up to 11 poses). Defaults to all. 

  Returns: 
    mesh loader, given as a LazyIterable
  """
  assert isinstance(simplify, Integral) or simplify is None, "If supplied, simplify must be an integer representing the desired number of triangles"
  import open3d as o3
  mesh_dir = _package_data("mesh_poses")

  ## Shape numbers. 0 means reference. 
  shape_nums = np.array([0,1,2,3,4,5,6,7,8,9,10]) if isinstance(shape_nums, str) and shape_nums == "all" else shape_nums
  shape_nums = np.array([i-1 if i != 0 else 10 for i in shape_nums])

  ## Choose the shapes
  _all_shapes = ["camel","face","horse","elephant","head","cat","flamingo","lion"]
  _default_shapes = ["camel", "cat", "elephant", "flamingo"]
  if isinstance(which, str) and which == "default":
    shape_types = _default_shapes
  elif isinstance(which, str) and which == "all":
    shape_types = _all_shapes
  else:
    assert all([s in _all_shapes for s in which]), "Invalid set of shapes given"
    shape_types = which

  ## Select the pose paths
  import re
  pose_dirs = [s+"-poses" for s in shape_types]
  get_pose_objs = lambda pd: np.array(list(filter(lambda s: s[-3:] == "obj", sorted(os.listdir(mesh_dir+"/"+pd)))))
  pose_objs = list(collapse([get_pose_objs(pose_type) for pose_type in pose_dirs]))
  pose_paths = []
  ptn = re.compile('(\w+)-(\d+)\.obj')
  for obj_name in pose_objs:
    m = ptn.match(obj_name)
    if m is not None and len(m.groups()) == 2 and int(m.groups()[1]) in shape_nums:
      pose_paths.append(mesh_dir + "/" + m.groups()[0]+"-poses" + "/" + obj_name)

  import networkx as nx
  from pbsig.itertools import LazyIterable
  from pbsig.pht import normalize_shape, stratify_sphere
  def load_mesh(i: int):
    mesh_in = o3.io.read_triangle_mesh(pose_paths[i])
    mesh_in.compute_vertex_normals()
    if simplify is not None: 
      mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=simplify)
      return mesh_smp
    # mesh_smp = mesh_in
    return mesh_in
    # mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=5000)
    # X = np.asarray(mesh_smp.vertices)
    # V = stratify_sphere(2, 64)
    # X_norm = normalize_shape(X, V=V, translate="hull")
    # return X_norm, mesh_smp
  meshes = LazyIterable(load_mesh, len(pose_paths))
  return meshes


class PatchData:
  def __init__(self, lazy: bool = False):
    from scipy.io import loadmat
    dct_basis = [None]*8
    dct_basis[0] = (1/np.sqrt(6)) * np.array([1,0,-1]*3).reshape(3,3)
    dct_basis[1] = dct_basis[0].T
    dct_basis[2] = (1/np.sqrt(54)) * np.array([1,-2,1]*3).reshape(3,3)
    dct_basis[3] = dct_basis[2].T
    dct_basis[4] = (1/np.sqrt(8)) * np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    dct_basis[5] = (1/np.sqrt(48)) * np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
    dct_basis[6] = (1/np.sqrt(48)) * np.array([[1,-2,1],[0,0,0],[-1,2,-1]])
    dct_basis[7] = (1/np.sqrt(216)) * np.array([[1,-2,-1],[-2,-4,-2],[1,-2,1]])
    self.dct_basis = np.hstack([np.ravel(b)[:,np.newaxis] for b in dct_basis])
    self.basis_lb = np.min(self.dct_basis)
    self.basis_ub = np.max(self.dct_basis)
    self.basis = "natural"
    self.lazy = lazy
    # self.patch_data = dct_patch_data['n50000Dct'] @ self.dct_basis.T

  def load_patch_data(self, basis: str = "natural"):
    assert basis == "natural" or basis == "dct", "Invalid basis supplied"
    dct_patch_data = loadmat("/Users/mpiekenbrock/Downloads/n50000Dct.mat")
    if basis == "natural":
      return dct_patch_data['n50000Dct'] @ self.dct_basis.T
    else: 
      return dct_patch_data['n50000Dct']

  @property
  def patch_data(self):
    if hasattr(self, "_patch_data") and self._patch_data[1] == self.basis:
      return self._patch_data[0]
    else:
      patch_data = self.load_patch_data(self.basis)
      if not self.lazy: 
        self._patch_data = (patch_data, self.basis)
      return patch_data

  def patch_rgba(self, patch: np.ndarray):
    """Returns an array of uint32 RGBa values normalizing the given patch to the range [0,255] """
    assert len(patch) == 9, "Invalid; must supply a 3x3 patch"
    normalize_unit = lambda x: (np.clip(x, self.basis_lb, self.basis_ub) - self.basis_lb)/(self.basis_ub - self.basis_lb)
    patch = normalize_unit(np.ravel(patch))
    img = np.empty((3,3), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((3,3,4))
    for cc, (i,j) in enumerate(product(range(3), range(3))):
      view[i, j, 0] = int(255 * patch[cc])
      view[i, j, 1] = int(255 * patch[cc])
      view[i, j, 2] = int(255 * patch[cc])
      view[i, j, 3] = 255
    return img 

  def figure_basis(self, **kwargs):
    """Constructs a row of figures showing the DCT basis vectors."""
    from bokeh.plotting import figure 
    from pbsig.vis import figure_plain
    from bokeh.layouts import row 
    fig_kwargs = dict(width=75, height=75) | kwargs
    basis_figs = [figure_plain(figure(**fig_kwargs)) for i in range(8)]
    for i, patch in enumerate(self.dct_basis.T):
      img = self.patch_rgba(patch)
      basis_figs[i].image_rgba(image=[img], x=-1, y=-1, dw=2, dh=2)
    return row(*basis_figs)

  def project_2d(self, indices: Union[str, Sequence] = "all") -> np.ndarray:
    """Projects the patch data onto 2-dimension using the DCT basis"""
    if isinstance(indices, str) and indices == "all":
      from pbsig.linalg import pca
      return pca(self.patch_data, 2)
    else: 
      assert len(indices) == 2
      return self.load_patch_data("natural") @ self.dct_basis[:,np.array(indices)]

  def figure_patches(self, coords: np.ndarray = None, patches: np.ndarray = None, size: float = 0.10, **kwargs):
    from bokeh.plotting import figure
    if coords is None and patches is None:
      from pbsig.shape import landmarks
      _basis = self.basis 
      self.basis = "natural"
      coords = self.project_2d()
      landmark_ind, _ = landmarks(coords, 15, seed=2971) # 2971 == closest to first basis vector, could also use 10009
      patches = self.patch_data[landmark_ind,:]
      coords = coords[landmark_ind,:]
      self.basis = _basis
    else:
      coords, patches = np.atleast_2d(coords), np.atleast_2d(patches)
    assert patches.shape[1] == 9, "'patches' must be 2-d array with 9 columns representing the patches"
    assert coords.shape[1] == 2 and len(coords) == len(patches), "Coordinates must be given for each patch"
    p = kwargs.get("figure", figure(width=300, height=300))
    imgs = [self.patch_rgba(patch) for patch in patches]
    half = size / 2.0
    p.image_rgba(image=imgs, x=coords[:,0]-half, y=coords[:,1]-half, dw=size, dh=size)
    return p 

  def figure_shade(self, X: np.ndarray = None, **kwargs):
    """Adds perceptively accurate scatter points 'X' to the supplied figure using datashader."""
    import datashader as ds
    import pandas as pd
    from bokeh.palettes import gray
    from bokeh.plotting import figure
    X = self.project_2d() if X is None else np.atleast_2d(X)
    assert X.ndim == 2, "X must be a two-dimensional point cloud"
    p = kwargs.get("figure", figure(width=300, height=300))
    canvas = ds.Canvas(plot_width=600, plot_height=600, x_range=(-1, 1), y_range=(-1, 1))
    patch_df = pd.DataFrame(dict(x=X[:,0], y=X[:,1]))
    agg = canvas.points(patch_df, x="x", y="y")
    im = ds.transfer_functions.shade(agg, cmap=gray(256), min_alpha=100, rescale_discrete_levels=False)
    # im = ds.transfer_functions.spread(im)
    im = ds.transfer_functions.dynspread(im, threshold=0.95)
    p.image_rgba(image=[im.to_numpy()], x=-1, y=-1, dw=2, dh=2, dilate=True)
    return p

  # p = figure(width=600, height=600)
