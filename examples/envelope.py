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
plt.scatter(roots, f(roots))


switch_types = np.sign(f.derivative(1)(roots))




# plt.gca().set_aspect('equal')