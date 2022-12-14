import numpy as np 
import matplotlib.pyplot as plt

from pbsig.utility import pairwise
import shapely
from shapely.geometry import GeometryCollection, Polygon, Point
from shapely import symmetric_difference, symmetric_difference_all

X = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
box = Polygon(X)
box2 = Polygon(X + np.array([[0.5, 0.25]]))

C = GeometryCollection([box, box2])
GeometryCollection([box, box.buffer(-0.5)])
box + box.buffer(1.5)


from shapely import minimum_bounding_circle
from shapely import minimum_clearance

minimum_clearance(box)

minimum_bounding_circle(box)


box - Point([0,0])
P = symmetric_difference(box, Point([0,0]))
symmetric_difference_all(symmetric_difference(box, Point([0,0])))
P.minimum_clearance

wo = symmetric_difference(box, Point([0,0]).buffer(0.01))
GeometryCollection([wo, minimum_bounding_circle(wo)])

GeometryCollection([box, Point([0,0]).buffer(0.01)]) 

from pbsig.color import bin_color

B = box.union(box2)
C = GeometryCollection([B, B.buffer(-0.5)])

X = np.random.uniform(size=(1500,2), low=-1.2, high=1.4)
# d = [distance(Point(x), B.boundary) for x in X]
d = [shortest_line(Point(x), B.boundary).length for x in X]

from shapely import get_coordinates
Y = get_coordinates(B.boundary)

plt.scatter(*X.T, c=bin_color(d))
plt.gca().set_aspect('equal')
plt.plot(*Y.T)



B.line_locate_point(Point(X[0,:]))
from shapely import LineString, Point

line = LineString([(0, 2), (0, 10)])
box.project(GeometryCollection([Point([0,0])]))

line.line_locate_point(Point([-0.4,0.1]))

GeometryCollection([line, Point([0,0])])
origin = Point([1,0])

line.project(origin)

from shapely import shortest_line


GeometryCollection([B, B.buffer(-1.12)])

# pip install shapely==1.8.5
B.buffer(-1.2).area

# modules = ['affinity', 'algorithms', 'coords', 'ctypes_declarations', 'errors', 'geometry', 'geos', 'impl', 'linref', 'predicates', 'speedups', 'topology']
# import importlib
# any(['minimum_bounding_circle' in dir(importlib.import_module("shapely."+mod_str)) for mod_str in modules])
  

# 'minimum_bounding_circle' in dir(shapely)

from shapely import minimum_bounding_circle

import shapely 
shapely.minimum_bounding_circle
from shapely.ops import minimum_bounding_circle

# import skgeom as sg

## Rectilinear partitioning 
from scipy.spatial import ConvexHull
from shapely import *
from shapely.geometry import *
#X = [[0,2],[0,4],[1,1],[1,2],[1,4],[2,0],[2,1],[2,4],[2,5],[3,0],[3,3],[3,4],[3,5],[4,0],[4,1],[4,3],[5,3],[5,4],[6,1],[6,2],[6,3],[6,4],[7,1],[7,2]]

p1 = Polygon([[0,2],[0,4],[1,4],[1,2]])
p2 = Polygon([[1,1],[1,2],[1,4],[2,4],[2,1]])
p3 = Polygon([[2,0],[2,1],[2,4],[2,5],[3,5],[3,4],[3,3],[3,0]])
p4 = Polygon([[3,0],[3,3],[4,3],[4,1],[4,0]])
p5 = Polygon([[4,1],[4,3],[5,3],[5,1]])
p6 = Polygon([[5,1],[5,4],[6,4],[6,1]])
p7 = Polygon([[6,1],[6,2],[7,2],[7,1]])
P = union_all([p1,p2,p3,p4,p5,p6,p7])
X = get_coordinates(P)

# GeometryCollection([union_all(R), union_all(R).buffer(-0.25),union_all(R).buffer(-0.5), union_all(R).buffer(-0.75), union_all(R).buffer(-1.0)])

## Generate random orthogonal rectangle
R = []
for i in range(10):
  x1,x2,y1,y2 = np.random.uniform(size=4, low=-5, high=5)
  r = np.array([[x1,y1],[x1,y2],[x2,y1],[x2,y2]])
  r = Polygon(r[ConvexHull(r).vertices,:])
  R.append(r)
P = union_all(R)
if isinstance(P, Polygon):
  print(len(list(P.interiors)))
P
partition_rectilinear(get_coordinates(P), False)



def search_outward(p, Q, direction="d"):
  """ Q is unique points, in clockwise order """
  assert direction in ['d', 'u', 'l', 'r']
  x,y = p
  outward_vertices = False
  if direction == "d":
    outward_vertices = np.logical_and(Q[:,0] == x, Q[:,1] < y)
  elif direction == "u":
    outward_vertices = np.logical_and(Q[:,0] == x, Q[:,1] > y)
  elif direction == "l":
    outward_vertices = np.logical_and(Q[:,0] < x, Q[:,1] == y)
  elif direction == "r":
    outward_vertices = np.logical_and(Q[:,0] > x, Q[:,1] == y)
  
  if any(outward_vertices):
    ## finds points incident to ray-casted edge, if they exist
    Q_ = Q[outward_vertices,:]
    dist = np.array([np.linalg.norm(q-p) for q in Q_])
    q = np.ravel(Q_[np.argmin(dist)])
  else:
    ## No such points exist; there must be a flat edge somewhere 
    edges = []
    for ((x1,y1), (x2,y2)) in cycle_window(Q):
      if direction == "d" or direction == "u":
        l,u = (x1, x2) if x1 < x2 else (x2, x1)
        if y1 == y2 and (l < x and x < u) and (y1 < y if direction == "d" else y < y1):
          edges.append(np.array([x,y1]))
      else: # left right
        l,u = (y1, y2) if y1 < y2 else (y2, y1)
        if x1 == x2 and (l < y and y < u) and (x1 < x if direction == "l" else x < x1):
          edges.append(np.array([x1,y]))
    if len(edges) == 0:
      raise ValueError("No outward edge exists")
    dist = np.array([np.linalg.norm(e-p) for e in edges])
    q = edges[np.argmin(dist)]
  return q, np.min(dist) ## (p,q) is the extended edge

## identify clockwise vs counter clockwise 
from pbsig.utility import cycle_window
import shapely
from shapely.geometry import GeometryCollection, LineString, Polygon
from shapely.ops import split

def _get_point_types(Q):
  " Determines for every point whether its convex, concave, flat, or unknown "
  ## cycle through coordinates in clockwise order
  CC = list(cycle_window(Q, 2, 3))
  CC = [CC[-1]] + CC[:-1]

  ## Collect convex/concave/flat point types
  pt = [] # point types
  for i, ((x1,y1),(x2,y2),(x3,y3)) in enumerate(CC):
    if x1 < x2 and x2 == x3 and y1 == y2 and y2 < y3:   # case 1
      pt.append((i, 1, 'd', 'r'))
    elif x1 == x2 and y1 < y2 and x2 < x3 and y2 == y3: # case 2
      pt.append((i, 2, 'n', 'n'))
    elif x1 < x2 and y1 == y2 and x2 == x3 and y2 > y3: # case 3
      pt.append((i, 2, 'n', 'n'))
    elif x1 == x2 and y1 > y2 and x2 > x3 and y2 == y3: # case 4
      pt.append((i, 2, 'n', 'n'))
    elif x1 > x2 and y1 == y2 and x2 == x3 and y2 < y3: # case 5
      pt.append((i, 2, 'n', 'n'))
    elif x1 == x2 and y1 < y2 and x2 > x3 and y2 == y3: # case 6
      pt.append((i, 1, 'u', 'r'))
    elif x1 == x2 and y1 > y2 and x2 < x3 and y2 == y3: # case 7
      pt.append((i, 1, 'l', 'd'))
    elif x1 > x2 and y1 == y2 and x2 == x3 and y2 > y3: # case 8 
      pt.append((i, 1, 'l', 'u'))
    elif (x1 == x2 and x2 == x3) or (y1 == y2 and y2 == y3):  # case 9 (flat)
      pt.append((i, 0, 'n', 'n'))
    else: 
      raise ValueError("unknown case")
      # pt.append((i, -1, 'n', 'n'))
  pt = np.array(pt, dtype=[('index', 'int'), ('type', 'int'),('d1', '<U21'),('d2','<U21')])
  return pt

def partition_rectilinear(X, ic: bool = False):
  """
  X := point coordinates defining the rectilinear polygon. Must be clockwise order. 
  """
  P = Polygon(X)
  assert P.is_valid and len(list(P.interiors)) == 0, 'X must be a simple polygon with no holes!' # ensure simple and hole-less polygon 
  assert np.allclose(np.array([min(d1,d2) for d1,d2 in zip(abs(np.diff(X[:,0])), abs(np.diff(X[:,1])))]), 0.0), "X must be rectilinear!"

  ## Map all the inputs to integer inputs 
  x_map = { x : i for i,x in enumerate(np.sort(np.unique(X[:,0]))) }
  y_map = { y : i for i,y in enumerate(np.sort(np.unique(X[:,1]))) }
  X = np.c_[[x_map[x] for x in X[:,0]], [y_map[y] for y in X[:,1]]]

  ## Input checking
  if all(X[-1,:] == X[0,:]):
    X_uniq = X[:-1,:]
  else:
    X_uniq = X.copy()
    X = np.vstack([X, X[0,:]])

  ## Quick check to see if something is a rectangle
  is_rect = lambda P: np.isclose(P.area, P.minimum_rotated_rectangle.area)
  
  ## Start off with entire point set + polygon
  Q = X_uniq.copy()
  B = Polygon(Q)  
  #[np.logical_or(pt['type'] == 1, pt['type'] == 2),:]

  ## Iteratively split the polygon into disjoint rectangular pieces
  R = []
  while not is_rect(B):
    pt = _get_point_types(Q)   
    Q_concave_pts = pt[pt['type'] == 1]
    min_dist, min_edge = np.inf, None
    for i, _, d1, d2 in Q_concave_pts:
      p = Q[i,:]
      q1, d_pq1 = search_outward(p, Q, d1)
      q2, d_pq2 = search_outward(p, Q, d2)
      if min(d_pq1, d_pq2) < min_dist:
        qb = q1 if d_pq1 < d_pq2 else q2
        min_edge = [p,qb]
        min_dist = min(d_pq1, d_pq2)
    # GeometryCollection([B, LineString(min_edge)])
    B1, B2 = split(B, LineString(min_edge)).geoms ## TODO: maybe move this into the above loop; guarentee breaking off piece is rectangular! 
    if is_rect(B1):
      R.append(B1)
      B = B2
    elif is_rect(B2):
      R.append(B2)
      B = B1
    else: 
      print("Invalid")
      break 
    Q = get_coordinates(B)[:-1]
    pt = _get_point_types(Q)   

  ## Cleanup by merging last rectangle back in, if possible 
  can_merge = [is_rect(union(B, r)) for r in R]
  if any(can_merge):
    i = np.flatnonzero(np.array(can_merge))[0]
    R[i] = union(R[i], B)
  else: 
    R = R + [B]
  
  ## Re-map integers coordinates back to original coordinates
  if not ic:  
    R_ = []
    x_inv, y_inv = np.array(list(x_map.keys())), np.array(list(y_map.keys()))
    for r in R:
      x_ = x_inv[get_coordinates(r)[:,0].astype(int)]
      y_ = y_inv[get_coordinates(r)[:,1].astype(int)]
      R_.append(Polygon(np.c_[x_, y_]))
    R = R_
  
  ## Return partitioned set of disjoint rectangles as a geometry collection 
  return GeometryCollection(R)



plt.plot(*Q.T)
plt.scatter(*Q.T, color=bin_color(np.arange(Q.shape[0])))
plt.gca().set_aspect('equal')


partition_rectilinear(X)

## sample the concave points, extend edges 
X_uniq = X[:-1,:]
# concave_pts = X_uniq[pt == 1,:]
# concave_pt = np.array([5,3]) # potential edge directions: down, right

## search down 



Polygon(X)
## Given two points (p,q), in clockwise order, determine the rectangle 
## in Q 
Polygon(Q)

plt.plot(*X[:-1,:].T)
plt.scatter(*X[:-1,:].T, color=bin_color(np.arange(X.shape[0]-1)))
from matplotlib.cm import cmaps_listed
cmaps_listed['viridis']

search_outward(np.array([5,3]), X_uniq, 'down')
search_down(np.array([6,2]), X_uniq)
search_down(np.array([3,3]), X_uniq)


plt.plot(*Q.T)
plt.scatter(*Q.T, color=bin_color(np.arange(Q.shape[0])))

search_outward(np.array([3,3]), Q, 'left')
search_outward(np.array([3,3]), Q, 'right')
search_outward(np.array([2,1]), Q, 'up')

search_down(np.array([5,3]), Q)
search_down(np.array([3,3]), Q)
search_down(np.array([6,2]), Q)
search_down(np.array([7,2]), Q)