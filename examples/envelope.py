from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

pts = np.random.uniform(low=0, high=1,size=(15,2))
pts = pts[np.argsort(pts[:,0]),:]
plt.plot(*pts.T)

x = np.linspace(0.0, 1.0, 20)
y = 0.50+np.cumsum(np.random.uniform(low=-0.10, high=0.10, size=20))

f = CubicSpline(x, y)

roots = list(filter(lambda x: x >= 0.0 and x <= 1.0, f.solve(0.30)))
dom = np.linspace(0, 1, 100)
plt.plot(dom, f(dom))
# plt.scatter(roots, f(roots))
plt.gca().set_xlim(0, 1)
plt.gca().set_ylim(-0.25, 1.25)

## Set the cutoff thres
threshold = np.min(y) + (np.max(y)-np.min(y))/2

## Partition a curve into intervals at some threshold
in_unit_interval = lambda x: x >= 0.0 and x <= 1.0
crossings = np.fromiter(filter(in_unit_interval, f.solve(threshold)), float)

## Draw cutoff + function
plt.plot(dom, f(dom))
plt.axhline(y=threshold, color='r', linestyle='-', linewidth=0.40)
plt.scatter(crossings, f(crossings))

## Determine the partitioning of the upper-envelope (1 indicates above threshold)
intervals = []

if len(crossings) == 0:
  is_above = f(0.50).item() >= threshold
  intervals.append((0.0, 1.0, 1 if is_above else 0))
else:
  if crossings[-1] != 1.0: 
    crossings = np.append(crossings, 1.0)
  b = 0.0
  df = f.derivative(1)
  df2 = f.derivative(2)
  for c in crossings:
    grad_sign = np.sign(df(c))
    if grad_sign == -1:
      intervals.append((b, c, 1))
    elif grad_sign == 1:
      intervals.append((b, c, 0))
    else: 
      accel_sign = np.sign(df2(c).item())
      if accel_sign > 0: # concave 
        intervals.append((b, c, 1))
      elif accel_sign < 0: 
        intervals.append((b, c, 0))
      else: 
        raise ValueError("Unable to detect")
    b = c


# plt.gca().set_aspect('equal')