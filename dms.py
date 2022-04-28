# dms.py 
# Contains a variety of functions for simulating dynamic metric spaces 
# Each function returns: 
# 1. a parameterized-closure which upon evaluation yields a new point set 'X' or set of pairwise distances
# 2. a dictionary containing information about the DMS, including e.g. parameter bounds, Betti numbers, etc.
import numpy as np
import autograd.numpy as auto_np 

def circle_family(n: int, lb: float = 0.0, ub: float = np.inf, sd: float = 0):
  theta = auto_np.linspace(0, 2*np.pi, n, endpoint=False)
  unit_circle = auto_np.c_[auto_np.cos(theta), auto_np.sin(theta)]
  unit_circle += auto_np.random.normal(scale=sd, size=unit_circle.shape)
  def circle(t: float):
    # t = auto_np.max([t, 0])
    return(auto_np.dot(unit_circle, auto_np.diag([t, t])))
  return(circle, { 'n_points': n,  'parameter_bounds' : [lb, ub] })


def moving_circles(n_circles: int, n_time_points: int):
  theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
  c = np.c_[np.cos(theta), np.sin(theta)] 
  centers = np.random.uniform(low=0, high=1, size=(n_circles,2))
  radii = np.random.uniform(low=0.10, high=0.35, size=n_circles)
  perturbations = np.random.normal(loc=0, scale=0.05, size=(n_circles, n_time_points))
  noise = np.random.uniform(low=-0.04, high=0.04, size=(n_circles, len(theta), 2))
  def f(t):
    assert 0 <= t and t <= 1, "Invalid t"
    t = t * n_time_points
    Ct = []
    for ci in range(n_circles):
      i, rem = divmod(t, 1)
      p_pos = np.sum(perturbations[ci, :int(i)])
      n_pos = np.sum(perturbations[ci, :int(i+1)])
      c_pos = (1-rem)*p_pos + rem*n_pos
      Ct.append(c*radii[int(ci)] + (centers[ci,:] + c_pos) + noise[ci,:])
    Ct = np.vstack(Ct)
    return(Ct)
  return(f, { 'n_points': n_circles*len(theta),  'parameter_bounds' : [0, 1] })

# def ellipse_family():


# import holoviews as hv
# import numpy as np

# hv.extension('bokeh')

# def radarray(N):
#     "Draw N random samples between 0 and 2pi radians"
#     return np.random.uniform(0, 2*np.pi, N)

# class BoidState(object):
#   def __init__(self, N=500, width=400, height=400):
#     self.width, self.height, self.iteration = width, height, 0
#     self.vel = np.vstack([np.cos(radarray(N)),  np.sin(radarray(N))]).T
#     r = min(width, height)/2*np.random.uniform(0, 1, N)
#     self.pos = np.vstack([width/2 +  np.cos(radarray(N))*r,  
#                           height/2 + np.sin(radarray(N))*r]).T

# def count(mask, n): 
#   return np.maximum(mask.sum(axis=1), 1).reshape(n, 1)

# def limit_acceleration(steer, n, maxacc=0.03):
#   norm = np.sqrt((steer*steer).sum(axis=1)).reshape(n, 1)
#   np.multiply(steer, maxacc/norm, out=steer, where=norm > maxacc)
#   return norm, steer

# class Boids(BoidState):
    
#   def flock(self, min_vel=0.5, max_vel=2.0):
#     n = len(self.pos)
#     dx = np.subtract.outer(self.pos[:,0], self.pos[:,0])
#     dy = np.subtract.outer(self.pos[:,1], self.pos[:,1])
#     dist = np.hypot(dx, dy)
#     mask_1, mask_2 = (dist > 0) * (dist < 25), (dist > 0) * (dist < 50)
#     target = np.dstack((dx, dy))
#     target = np.divide(target, dist.reshape(n,n,1)**2, out=target, where=dist.reshape(n,n,1) != 0)
#     steer = (target*mask_1.reshape(n, n, 1)).sum(axis=1) / count(mask_1, n)
#     norm = np.sqrt((steer*steer).sum(axis=1)).reshape(n, 1)
#     steer = max_vel*np.divide(steer, norm, out=steer, where=norm != 0) - self.vel
#     norm, separation = limit_acceleration(steer, n)
#     target = np.dot(mask_2, self.vel)/count(mask_2, n)
#     norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
#     target = max_vel * np.divide(target, norm, out=target, where=norm != 0)
#     steer = target - self.vel
#     norm, alignment = limit_acceleration(steer, n)
#     target = np.dot(mask_2, self.pos)/ count(mask_2, n)
#     desired = target - self.pos
#     norm = np.sqrt((desired*desired).sum(axis=1)).reshape(n, 1)
#     desired *= max_vel / norm
#     steer = desired - self.vel
#     norm, cohesion = limit_acceleration(steer, n)
#     self.vel += 1.5 * separation + alignment + cohesion
#     norm = np.sqrt((self.vel*self.vel).sum(axis=1)).reshape(n, 1)
#     np.multiply(self.vel, max_vel/norm, out=self.vel, where=norm > max_vel)
#     np.multiply(self.vel, min_vel/norm, out=self.vel, where=norm < min_vel)
#     self.pos += self.vel + (self.width, self.height)
#     self.pos %= (self.width, self.height)
#     self.iteration += 1

#%opts VectorField [xaxis=None yaxis=None] (scale=0.08)
#%opts VectorField [normalize_lengths=False rescale_lengths=False] 

# boids = Boids(500)

# for i in range(1000):
#   boids.flock()
#   plt.scatter(*boids.pos.T)


# def boids_vectorfield(boids, iteration=1):
#   angle = (np.arctan2(boids.vel[:, 1], boids.vel[:, 0]))
#   return hv.VectorField([
#     boids.pos[:,0], boids.pos[:,1], 
#     angle, np.ones(boids.pos[:,0].shape)], extents=(0,0,400,400), 
#     label='Iteration: %s' % boids.iteration
#   )

# boids_vectorfield(boids, iteration=50)

# from holoviews.streams import Stream

# def flock():
#   boids.flock()
#   return boids_vectorfield(boids)

# dmap = hv.DynamicMap(flock, streams=[Stream.define('Next')()])
# dmap
# dmap.periodic(0.01, timeout=60, block=True)