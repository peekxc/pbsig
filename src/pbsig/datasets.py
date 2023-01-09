import os
import numpy as np
import importlib.resources as pkg_resources
 
from scipy.spatial.distance import pdist, cdist, squareform
from PIL import Image, ImageDraw, ImageFont
from typing import * 
from numpy.typing import ArrayLike
from pathlib import Path

# Package relative imports 
from .utility import *

# def random_graph(n: int, p: float):
from .__init__ import _package_data


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
  from pbsig import data as package_data_mod
  data_path = package_data_mod.__path__._path[0]
  fonts = ['lato-bold','opensans','ostrich', 'oswald','roboto']
  fonts_fn = ["Lato-Bold", "OpenSans-Bold", "ostrich-regular", "Oswald-Bold", "Roboto-Bold"]
  if font == ["Lato-Bold", "OpenSans", "Ostrich", "Oswald", "Roboto"]:
    font_path = data_path + '/Lato-Bold.ttf'
  elif isinstance(font, str) and font.lower() in fonts:
    font_path = data_path +'/' + fonts_fn[fonts.index(font.lower())] + '.ttf'
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

def freundenthal_image(image: ArrayLike, threshold: int = 200):
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
  V = scale_diameter(v_pos - center) 
  return(V, E, T)

# def letters():
#   from pbsig import data as package_data_mod
  #   data_path = package_data_mod.__path__._path[0]

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
  
def mpeg7(contour: bool = True, simplify: int = 150):
  import io
  # from . import *
  from os.path import exists
  from pbsig import data as package_data_mod
  from pbsig.utility import simplify_outline
  from PIL import Image, ImageFilter
  #base_dir = package_data_mod.__path__._path[0] + '/mpeg7'
  base_dir = _package_data('mpeg7')
  mpeg7 = []
  # for fn in os.listdir(base_dir):
  #   if fn[-3:] == 'gif':
  #     mpeg7.append(Image.open(base_dir+'/'+fn))
  # for member in open_tar(base_dir+"/mpeg7.tar.xz", mode="r:xz"):
  #   f = tar.extractfile(member)
  #   if member.size > 0:
  #     content = f.read()
  #     mpeg7.append(Image.open(io.BytesIO(content)))
  # med types: "bird", "bell"
  # bad types: "lizzard", "chicken"
  shape_types = ["turtle", "watch", "bird", "bone", "bell", "bat", "beetle", "butterly", "car", "cup", "dog"]
  shape_nums = [1,2,3,4,5,6,7,8] #3,4,5
  normalize = lambda X: (X - np.min(X))/(np.max(X)-np.min(X))*255
  dataset = {}
  for st in shape_types:
    for sn in shape_nums:
      img_dir = base_dir + f"/{st}-{sn}.gif"
      assert exists(img_dir), "Image not found"
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
  