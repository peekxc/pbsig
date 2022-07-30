import numpy as np
import importlib.resources as pkg_resources

from scipy.spatial.distance import pdist, cdist, squareform
from PIL import Image, ImageDraw, ImageFont
from typing import * 
from numpy.typing import ArrayLike
from pathlib import Path

# Package relative imports 
from .utility import *

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

def letters():
  from pbsig import data as package_data_mod
  data_path = package_data_mod.__path__._path[0]

  
  return(V, E, T)

