import os
import io
import numpy as np
import pickle 

from os.path import exists
from scipy.spatial.distance import pdist, cdist, squareform
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import * 
from numpy.typing import ArrayLike
from pathlib import Path
from splex import filtration

# Package relative imports 
from .utility import *
from .combinatorial import *
from .simplicial import *
# from .__init__ import _package_data

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


# class FiltrationFamily:
#   """ Constructs a parameterized family of filtration-like objects derived from a fixed simplicial complex. """

#   def __init__(self):
#     pass

#   def __call__(self, bw: float):
#     pass

# class DensityFiltration(FiltrationFamily):
#   """ Constructs a bandwidth-parameterized family of density-filtered complexes """
#   def __init__(self, X: ArrayLike):
#     pass

#   def __call__(self, bw: float):    
#     self.K
#     return self.K 

#   def __iter__(self):
#     for bw in self.bandwidths:
#       yield self.K


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
  noise = _reject_sampler(n_noise, r)
  X = np.vstack((circle, noise))
  return X 

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
  """
  Generates a letter image 
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


def mpeg7(contour: bool = True, simplify: int = 150, which: str = 'default'):
  base_dir = _package_data('mpeg7')
  if simplify == 150 and contour == True and which == "default":
    mpeg7 = pickle.load(open(base_dir + "/mpeg_small.pickle", "rb"))
    return mpeg7
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
  default_shapes = [
    "turtle", "watch", "bird", "bone", "bell", "bat", "beetle", "butterly", 
    "car", "cup", "teddy", "spoon", "shoe", "ray", "sea_snake", "personal_car", 
    "key", "horse", "hammer", "frog", "fork", "flatfish", "elephant"
  ]
  shape_types = default_shapes if which == "default" else _all_shapes
  shape_nums = range(20) #[1,2,3,4,5,6,7,8,9,10,11,12] #3,4,5
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
        dataset[(st, sn)] = np.array([l.start for l in S]) 
      else: 
        dataset[(st, sn)] = img_gray
  return(dataset)
  