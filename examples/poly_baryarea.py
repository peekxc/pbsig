import numpy as np 
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from pprint import pprint

k = np.sqrt(2)
P = Polygon([[-1,0], [1,0], [0,k]])
t1 = np.array([0,0])
t2 = 0.5*np.array([0,k]) + 0.5*np.array([1,0])
t3 = 0.5*np.array([0,k]) + 0.5*np.array([-1,0])
c = np.array(list(P.centroid.coords)[0])

P1 = Polygon([[0,0], c, t2, [1,0]])
P2 = Polygon([[0,k], t3, c, t2])
P3 = Polygon([[0,0], c, t3, [-1,0]])
# MultiPolygon([P1, P2, P3])

coords = lambda x: np.array(list(x)[0])

C = np.vstack([
  coords(P1.centroid.coords),
  coords(P2.centroid.coords),
  coords(P3.centroid.coords)
])

triangle_p = MultiPolygon([P1, P2, P3])
tri_points = MultiPoint([P1.centroid, P2.centroid, P3.centroid])
#pprint(triangle_p)

subdivision_centroid = (1/P.area)*(P1.area*C[0,:] + P2.area*C[1,:]  + P3.area*C[2,:])

np.vstack([[-1,0], [1,0], [0,k]]).mean(axis=0)
coords(P.centroid.coords)

S = Polygon([[-1,0], [1,0], [1,1], [-1,1]])

def make_n_gon(sides, radius=1, rotation=0, translation=None):
  from math import pi, sin, cos
  s = pi * 2 / sides # segment
  points = [(sin(s * i + rotation) * radius, cos(s * i + rotation) * radius) for i in range(sides)]
  if translation:
    points = [[sum(pair) for pair in zip(point, translation)] for point in points]
  return(np.array(points))


from pbsig.utility import simplify_outline
import matplotlib.pyplot as plt
from pbsig.pht import rotate_S1
theta = np.linspace(0, 2*np.pi, 1500, endpoint=False)

for n in range(3, 15):
  S10 = np.array([l.start for l in simplify_outline(make_n_gon(n), 1000)])
  S10_dgm = np.vstack([(min(f), max(f)) for f in rotate_S1(S10, 1500, include_direction=False)])
  plt.plot(*S10_dgm.T, linewidth=0.75, label=f"{n}-gon vine")
plt.legend()

from shapely.ops import voronoi_diagram
S3 = np.array([l.start for l in simplify_outline(make_n_gon(3), 10)])
vd = voronoi_diagram(Polygon(S3))

Polygon(S3).union(vd)



