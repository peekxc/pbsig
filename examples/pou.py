import numpy as np 
import matplotlib.pyplot as plt

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

## identify clockwise vs counter clockwise 
from pbsig.utility import cycle_window

# clockwise coordinates
CC = list(cycle_window(X[:-1], 2, 3))
CC = [CC[-1]] + CC[:-1]

for i, ((x1,y1),(x2,y2),(x3,y3)) in enumerate(CC):
  if x1 < x2 and x2 == x3 and y1 == y2 and y2 > y3:
    print(f"{i} is concave")
  elif x1 == x2 and x2 < x3 and y1 > y2 and y2 == y3:
    print(f"{i} is convex")
  elif x1 < x2 and y1 == y2 and x2 == x3 and y2 < y3:
    print(f"{i} is convex")
  elif x1 == x2 and y1 < y2 and x2 < x3 and y2 == y3:
    print(f"{i} is convex")
  elif 

x = 
